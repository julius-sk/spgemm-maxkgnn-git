#!/usr/bin/env python3
"""
Modified SAGE Model with MaxK SpGEMM Integration
Integrates MaxK CUDA kernels into DGL SAGE for accelerated training
"""

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from torch.autograd import Function
import math

# Import our MaxK SpGEMM function
from maxk_spgemm_function import MaxKSpmmWrapper, maxk_spgemm, MAXK_KERNELS_AVAILABLE

class MaxK(Function):
    """MaxK activation function from original code"""
    @staticmethod
    def forward(ctx, input, k=1):
        topk, indices = input.topk(k, dim=1)
        mask = torch.zeros_like(input)
        mask.scatter_(1, indices, 1)
        output = input * mask
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None

class MaxKSAGEConv(nn.Module):
    """
    SAGE Convolution layer with MaxK SpGEMM acceleration
    Replaces DGL's SAGEConv with direct MaxK kernel calls
    """
    
    def __init__(self, in_feats, out_feats, aggregator_type='mean', 
                 feat_drop=0., bias=True, norm=None, activation=None, k_value=32):
        super(MaxKSAGEConv, self).__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggregator_type = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.k_value = k_value
        
        # Linear transformations
        self.fc_self = Linear(in_feats, out_feats, bias=False)
        self.fc_neigh = Linear(in_feats, out_feats, bias=bias)
        
        # Normalization
        self.norm = norm
        
        # MaxK SpGEMM wrapper
        self.maxk_wrapper = MaxKSpmmWrapper()
        
        # Graph metadata (will be set during first forward pass)
        self.graph_indices = None
        self.graph_values = None
        self.graph_indptr = None
        self.metadata_loaded = False
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def set_graph_data(self, graph, graph_name=""):
        """
        Set graph data for MaxK kernel usage
        
        Args:
            graph: DGL graph
            graph_name: Name for loading warp4 metadata
        """
        # Extract CSR format from DGL graph
        graph = graph.local_var()
        
        # Get CSR representation
        indptr, indices, _ = graph.adj_tensors('csr')
        
        # Store graph data
        self.graph_indices = indices.int()
        self.graph_indptr = indptr.int()
        
        # Create uniform edge weights (can be modified for weighted graphs)
        num_edges = indices.size(0)
        self.graph_values = torch.ones(num_edges, device=indices.device, dtype=torch.float32)
        
        # Load MaxK metadata if kernels are available
        if MAXK_KERNELS_AVAILABLE and graph_name:
            self.metadata_loaded = self.maxk_wrapper.load_metadata(graph_name)
            if self.metadata_loaded:
                print(f"✅ MaxK metadata loaded for {graph_name}")
            else:
                print(f"⚠️ Using fallback for {graph_name}")
        else:
            print("⚠️ MaxK kernels not available, using cuSPARSE fallback")
            self.metadata_loaded = False
    
    def forward(self, graph, feat):
        """
        Forward pass with MaxK SpGEMM acceleration
        
        Args:
            graph: DGL graph
            feat: Input node features
            
        Returns:
            Output node features
        """
        with graph.local_scope():
            # Apply dropout
            feat = self.feat_drop(feat)
            
            # Self connection
            h_self = self.fc_self(feat)
            
            # Neighbor aggregation using MaxK SpGEMM
            if self.graph_indices is not None and MAXK_KERNELS_AVAILABLE:
                # Use MaxK SpGEMM for neighbor aggregation
                try:
                    # Apply linear transformation before aggregation
                    feat_neigh = self.fc_neigh(feat)
                    
                    # MaxK SpGEMM aggregation
                    h_neigh = self.maxk_wrapper.spmm(
                        self.graph_indices,
                        self.graph_values,
                        feat_neigh,
                        self.k_value,
                        self.graph_indptr
                    )
                    
                except Exception as e:
                    print(f"⚠️ MaxK kernel failed: {e}, falling back to DGL")
                    # Fallback to DGL implementation
                    graph.ndata['h'] = self.fc_neigh(feat)
                    graph.update_all(dgl.function.copy_u('h', 'm'), 
                                   dgl.function.mean('m', 'h_neigh'))
                    h_neigh = graph.ndata['h_neigh']
            else:
                # Fallback to DGL implementation
                graph.ndata['h'] = self.fc_neigh(feat)
                if self.aggregator_type == 'mean':
                    graph.update_all(dgl.function.copy_u('h', 'm'), 
                                   dgl.function.mean('m', 'h_neigh'))
                elif self.aggregator_type == 'sum':
                    graph.update_all(dgl.function.copy_u('h', 'm'), 
                                   dgl.function.sum('m', 'h_neigh'))
                else:
                    raise ValueError(f"Unsupported aggregator: {self.aggregator_type}")
                h_neigh = graph.ndata['h_neigh']
            
            # Combine self and neighbor representations
            h = h_self + h_neigh
            
            # Apply normalization
            if self.norm is not None:
                h = self.norm(h)
            
            # Apply activation
            if self.activation is not None:
                h = self.activation(h)
            
            return h

class MaxKSAGE(nn.Module):
    """
    SAGE model with MaxK SpGEMM acceleration
    Modified version of the original SAGE class
    """
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk", graph_name=""):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_hid_layers
        self.graph_name = graph_name
        
        # Build MaxK SAGE layers
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
            else:
                norm_layer = None
            
            # Use our custom MaxKSAGEConv instead of DGL's SAGEConv
            layer = MaxKSAGEConv(
                in_feats=hid_size,
                out_feats=hid_size,
                aggregator_type='mean',
                feat_drop=feat_drop,
                norm=norm_layer,
                k_value=maxk
            )
            self.layers.append(layer)
        
        # Input and output linear layers
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        
        # MaxK activation functions
        for i in range(self.num_layers):
            exec(f"self.maxk{i} = MaxK.apply")
            exec(f"self.k{i} = maxk")
        
        self.nonlinear = nonlinear
        self.graph_set = False
    
    def set_graph(self, graph):
        """Set graph data for all MaxK layers"""
        if not self.graph_set:
            for layer in self.layers:
                if isinstance(layer, MaxKSAGEConv):
                    layer.set_graph_data(graph, self.graph_name)
            self.graph_set = True
            print(f"✅ Graph data set for MaxK-SAGE model")
    
    def forward(self, g, x):
        """Forward pass with MaxK acceleration"""
        # Set graph data on first forward pass
        if not self.graph_set:
            self.set_graph(g)
        
        # Input transformation
        x = self.lin_in(x)
        
        # Hidden layers with MaxK activation and SpGEMM
        for i in range(self.num_layers):
            # Apply MaxK activation if specified
            if self.nonlinear == 'maxk':
                x = eval(f"self.maxk{i}(x, self.k{i})")
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            
            # Apply MaxK SAGE convolution
            x = self.layers[i](g, x)
        
        # Output transformation
        x = self.lin_out(x)
        
        return x

# Keep original models for compatibility, but add MaxK versions
class SAGE(nn.Module):
    """Original SAGE model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_hid_layers
        # Multi-layers SAGEConv
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
                # norm_layer = nn.BatchNorm1d(hid_size)
            else:
                norm_layer = None
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean", feat_drop=feat_drop, norm=norm_layer))
        # self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean", feat_drop=feat_drop))

        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        for i in range(self.num_layers):
            exec("self.maxk{} = MaxK.apply".format(i))
            exec("self.k{} = maxk".format(i))

        self.nonlinear = nonlinear
    def forward(self, g, x):
        x = self.lin_in(x)

        for i in range(self.num_layers):
            if self.nonlinear == 'maxk':
                x = eval("self.maxk{}(x, self.k{})".format(i, i))
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            # x = self.dropout(x)
            x = self.layers[i](g, x)
        x = self.lin_out(x)

        return x

class GCN(nn.Module):
    """Original GCN model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        
        # two-layer GCN
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.GraphConv(hid_size, hid_size, activation=None, weight=None))
            if self.norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
                # self.normlayers.append(nn.BatchNorm1d(hid_size))

        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
        for i in range(self.num_layers):
            init.xavier_uniform_(self.linlayers[i].weight)
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)



        self.nonlinear = nonlinear
        for i in range(self.num_layers):
            exec("self.maxk{} = MaxK.apply".format(i))
            exec("self.k{} = maxk".format(i))

    def forward(self, g, x):
        x = self.lin_in(x).relu()

        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            if self.nonlinear == 'maxk':
                x = eval("self.maxk{}(x, self.k{})".format(i, i))
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            x = self.dropoutlayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)
        x = self.lin_out(x)
        return x

class GIN(nn.Module):
    """Original GIN model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        
        # two-layer GCN
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.pytorch.conv.GINConv(learn_eps=True, activation=None))
            if self.norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
                # self.normlayers.append(nn.BatchNorm1d(hid_size))

        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
        for i in range(self.num_layers):
            init.xavier_uniform_(self.linlayers[i].weight)
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)



        self.nonlinear = nonlinear
        for i in range(self.num_layers):
            exec("self.maxk{} = MaxK.apply".format(i))
            exec("self.k{} = maxk".format(i))

    def forward(self, g, x):
        x = self.lin_in(x).relu()

        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            if self.nonlinear == 'maxk':
                x = eval("self.maxk{}(x, self.k{})".format(i, i))
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            x = self.dropoutlayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)
        x = self.lin_out(x)
        return x
    
class GNN_res(nn.Module):
    """Original GNN_res model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.dropoutlayers1 = nn.ModuleList()
        self.dropoutlayers2 = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        
        # two-layer GCN
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers1.append(nn.Dropout(feat_drop))
            self.dropoutlayers2.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.GraphConv(hid_size, hid_size, activation=None, weight=None))
            if self.norm:
                # self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
                self.normlayers.append(nn.BatchNorm1d(hid_size))

        self.linlayers1 = nn.ModuleList()
        self.linlayers2 = nn.ModuleList()
        self.reslayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers1.append(Linear(hid_size, hid_size))
            self.linlayers2.append(Linear(hid_size, hid_size))
            self.reslayers.append(Linear(hid_size, hid_size))
        for i in range(self.num_layers):
            init.xavier_uniform_(self.linlayers1[i].weight)
            init.xavier_uniform_(self.linlayers2[i].weight)
            init.xavier_uniform_(self.reslayers[i].weight)
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)

    def forward(self, g, x):
        x = self.lin_in(x).relu()

        for i in range(self.num_layers):
            x_res = self.reslayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)

            x = self.linlayers1[i](x)
            x = F.relu(x)
            x = self.dropoutlayers1[i](x)
            x = self.linlayers2[i](x)
            
            x = x_res + x
            x = F.relu(x)
            x = self.dropoutlayers2[i](x)

        x = self.lin_out(x)
        return x

def test_maxk_sage():
    """Test the MaxK SAGE implementation"""
    print("🧪 Testing MaxK SAGE Model")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Create test graph and features
    num_nodes = 1000
    num_edges = 5000
    feat_dim = 128
    hidden_dim = 64
    output_dim = 10
    
    # Create random graph
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g = dgl.graph((src, dst), num_nodes=num_nodes).to('cuda')
    
    # Create features
    features = torch.randn(num_nodes, feat_dim).cuda()
    
    # Test MaxK SAGE
    print("🔄 Testing MaxK SAGE...")
    maxk_model = MaxKSAGE(
        in_size=feat_dim,
        hid_size=hidden_dim,
        num_hid_layers=2,
        out_size=output_dim,
        maxk=32,
        graph_name="test_graph"
    ).cuda()
    
    # Forward pass
    try:
        output = maxk_model(g, features)
        print(f"✅ MaxK SAGE forward pass: {features.shape} -> {output.shape}")
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        print("✅ MaxK SAGE backward pass successful")
        
        # Compare with original SAGE
        print("🔄 Comparing with original SAGE...")
        original_model = SAGE(
            in_size=feat_dim,
            hid_size=hidden_dim,
            num_hid_layers=2,
            out_size=output_dim,
            maxk=32
        ).cuda()
        
        original_output = original_model(g, features)
        print(f"✅ Original SAGE: {features.shape} -> {original_output.shape}")
        
        print("🎉 MaxK SAGE integration test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_maxk_sage()
