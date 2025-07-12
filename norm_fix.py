#!/usr/bin/env python3
"""
FIXED: MaxK SAGE Implementation with Proper Mean Normalization
Addresses the missing normalization in MaxK CUDA kernels that was causing exploding gradients
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
try:
    from maxk_spgemm_function import MaxKSpmmWrapper, maxk_spgemm, MAXK_KERNELS_AVAILABLE
    print("✅ MaxK CUDA kernels loaded for training integration")
except ImportError:
    MAXK_KERNELS_AVAILABLE = False
    print("⚠️ MaxK CUDA kernels not available, falling back to DGL")

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
    FIXED: SAGE Convolution layer with MaxK SpGEMM acceleration and proper normalization
    Now includes degree-based mean normalization to match DGL's SAGEConv behavior
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
        if MAXK_KERNELS_AVAILABLE:
            self.maxk_wrapper = MaxKSpmmWrapper()
        else:
            self.maxk_wrapper = None
        
        # Graph metadata (will be set during first forward pass)
        self.graph_indices = None
        self.graph_values = None
        self.graph_indptr = None
        self.node_degrees = None  # ADDED: Store node degrees for normalization
        self.metadata_loaded = False
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def set_graph_data(self, graph, graph_name=""):
        """
        Set graph data for MaxK kernel usage with degree computation
        
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
        
        # FIXED: Compute and store node degrees for proper normalization
        self.node_degrees = graph.in_degrees().float().to(indices.device)
        # Avoid division by zero for isolated nodes
        self.node_degrees = torch.clamp(self.node_degrees, min=1.0)
        
        print(f"📊 Graph stats: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
        print(f"📊 Degree stats: min={self.node_degrees.min():.1f}, max={self.node_degrees.max():.1f}, avg={self.node_degrees.mean():.1f}")
        
        # Load MaxK metadata if kernels are available
        if MAXK_KERNELS_AVAILABLE and graph_name and self.maxk_wrapper:
            self.metadata_loaded = self.maxk_wrapper.load_metadata(graph_name)
            if self.metadata_loaded:
                print(f"✅ MaxK metadata loaded for {graph_name}")
            else:
                print(f"⚠️ Using fallback for {graph_name}")
        else:
            print("⚠️ MaxK kernels not available, using DGL fallback")
            self.metadata_loaded = False
    
    def forward(self, graph, feat):
        """
        FIXED: Forward pass with MaxK SpGEMM acceleration and proper mean normalization
        
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
            
            # Neighbor aggregation using MaxK SpGEMM with proper normalization
            if (self.graph_indices is not None and 
                MAXK_KERNELS_AVAILABLE and 
                self.maxk_wrapper and 
                self.metadata_loaded):
                
                try:
                    # Apply linear transformation before aggregation
                    feat_neigh = self.fc_neigh(feat)
                    
                    # MaxK SpGEMM aggregation (this gives us the SUM of neighbor features)
                    h_neigh_sum = self.maxk_wrapper.spmm(
                        self.graph_indices,
                        self.graph_values,
                        feat_neigh,
                        self.k_value,
                        self.graph_indptr
                    )
                    
                    # FIXED: Apply mean normalization by dividing by node degrees
                    # This is what was missing and causing the exploding gradients!
                    h_neigh = h_neigh_sum / self.node_degrees.unsqueeze(-1)
                    
                    print(f"🔧 Applied MaxK kernel with mean normalization")
                    print(f"   Before norm: min={h_neigh_sum.min():.4f}, max={h_neigh_sum.max():.4f}")
                    print(f"   After norm:  min={h_neigh.min():.4f}, max={h_neigh.max():.4f}")
                    
                except Exception as e:
                    print(f"⚠️ MaxK kernel failed: {e}, falling back to DGL")
                    # Fallback to DGL implementation
                    graph.ndata['h'] = self.fc_neigh(feat)
                    graph.update_all(dgl.function.copy_u('h', 'm'), 
                                   dgl.function.mean('m', 'h_neigh'))
                    h_neigh = graph.ndata['h_neigh']
            else:
                # Fallback to DGL implementation (already includes mean normalization)
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
    FIXED: SAGE model with MaxK SpGEMM acceleration and proper normalization
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
            
            # Use our custom MaxKSAGEConv with normalization fix
            layer = MaxKSAGEConv(
                in_feats=hid_size,
                out_feats=hid_size,
                aggregator_type='mean',  # Ensure mean aggregation
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
            print(f"✅ Graph data set for MaxK-SAGE model with normalization")
    
    def forward(self, g, x):
        """Forward pass with MaxK acceleration and proper normalization"""
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
            
            # Apply MaxK SAGE convolution (now with proper normalization)
            x = self.layers[i](g, x)
        
        # Output transformation
        x = self.lin_out(x)
        
        return x

# Alternative approach: Use DGL's SAGEConv with MaxK activation only
class HybridMaxKSAGE(nn.Module):
    """
    ALTERNATIVE: Hybrid approach that uses DGL's proven SAGEConv for message passing
    but applies MaxK activation. This guarantees correct normalization.
    """
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_hid_layers
        
        # Use DGL's proven SAGEConv layers (they handle normalization correctly)
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
            else:
                norm_layer = None
            
            # Use DGL's SAGEConv which includes proper mean normalization
            self.layers.append(dglnn.SAGEConv(
                hid_size, hid_size, "mean", 
                feat_drop=feat_drop, 
                norm=norm_layer
            ))
        
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
    
    def forward(self, g, x):
        """Forward pass using DGL's SAGEConv with MaxK activation"""
        # Input transformation
        x = self.lin_in(x)
        
        # Hidden layers with MaxK activation and DGL's message passing
        for i in range(self.num_layers):
            # Apply MaxK activation if specified
            if self.nonlinear == 'maxk':
                x = eval(f"self.maxk{i}(x, self.k{i})")
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            
            # Use DGL's SAGEConv (includes proper mean normalization)
            x = self.layers[i](g, x)
        
        # Output transformation
        x = self.lin_out(x)
        
        return x

# Keep original models for compatibility
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

def test_normalization_fix():
    """Test the normalization fix"""
    print("🧪 Testing MaxK SAGE Normalization Fix")
    print("=" * 50)
    
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
    
    print(f"📊 Test graph: {num_nodes} nodes, {num_edges} edges")
    print(f"📊 Average degree: {num_edges * 2 / num_nodes:.1f}")
    
    # Test different approaches
    models = {
        "Original SAGE": SAGE(feat_dim, hidden_dim, 2, output_dim, maxk=32),
        "Hybrid MaxK-SAGE": HybridMaxKSAGE(feat_dim, hidden_dim, 2, output_dim, maxk=32),
    }
    
    if MAXK_KERNELS_AVAILABLE:
        models["Fixed MaxK-SAGE"] = MaxKSAGE(feat_dim, hidden_dim, 2, output_dim, maxk=32, graph_name="test")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Testing {name}...")
        model = model.cuda()
        
        try:
            # Forward pass
            output = model(g, features)
            
            # Check for reasonable output ranges
            output_min, output_max = output.min().item(), output.max().item()
            output_mean = output.mean().item()
            output_std = output.std().item()
            
            print(f"✅ {name} forward pass successful")
            print(f"   Output range: [{output_min:.4f}, {output_max:.4f}]")
            print(f"   Output mean: {output_mean:.4f}, std: {output_std:.4f}")
            
            # Check if values are reasonable (not exploding)
            if abs(output_max) < 1000 and abs(output_min) < 1000:
                print(f"✅ {name} produces reasonable output values")
            else:
                print(f"⚠️ {name} may have exploding values")
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            print(f"✅ {name} backward pass successful")
            
            results[name] = {
                'output_range': (output_min, output_max),
                'output_mean': output_mean,
                'output_std': output_std,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Compare results
    print(f"\n📊 Comparison Summary:")
    print("=" * 50)
    for name, result in results.items():
        if result['success']:
            print(f"{name:20s}: range=[{result['output_range'][0]:8.4f}, {result['output_range'][1]:8.4f}], "
                  f"mean={result['output_mean']:8.4f}")
        else:
            print(f"{name:20s}: FAILED - {result['error']}")
    
    print(f"\n💡 Key Points:")
    print("- Original SAGE should work correctly (baseline)")
    print("- Hybrid MaxK-SAGE combines MaxK activation with DGL's proven message passing")
    print("- Fixed MaxK-SAGE includes degree normalization in the CUDA kernel")
    print("- All approaches should produce similar output ranges")

if __name__ == "__main__":
    test_normalization_fix()





#!/usr/bin/env python3
"""
Complete DGL-Equivalent MaxK SAGE Implementation
Mathematically identical to DGL SAGEConv with MaxK kernel acceleration
Addresses ALL differences: normalization, transformation timing, edge weights, etc.
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
from dgl import function as fn
from dgl.utils import check_eq_shape, expand_as_pair

# Import our MaxK SpGEMM function
try:
    from maxk_spgemm_function import MaxKSpmmWrapper, maxk_spgemm, MAXK_KERNELS_AVAILABLE
    print("✅ MaxK CUDA kernels loaded for training integration")
except ImportError:
    MAXK_KERNELS_AVAILABLE = False
    print("⚠️ MaxK CUDA kernels not available, falling back to DGL")

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

class ExactDGLEquivalentMaxKSAGEConv(nn.Module):
    """
    COMPLETE DGL-Equivalent MaxK SAGE Convolution
    Mathematically identical to DGL's SAGEConv with MaxK kernel acceleration
    Implements ALL DGL features: lin_before_mp, edge weights, bipartite graphs, etc.
    """
    
    def __init__(self, in_feats, out_feats, aggregator_type='mean', 
                 feat_drop=0., bias=True, norm=None, activation=None, k_value=32):
        super(ExactDGLEquivalentMaxKSAGEConv, self).__init__()
        
        # Validate aggregator type (exactly like DGL)
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise ValueError(
                f"Invalid aggregator_type. Must be one of {valid_aggre_types}. "
                f"But got {aggregator_type!r} instead."
            )
        
        # Store parameters exactly like DGL
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.k_value = k_value
        
        # Aggregator-specific layers (exactly like DGL)
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )
        
        # Main linear layers (exactly like DGL)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        
        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)
        
        # MaxK SpGEMM wrapper
        if MAXK_KERNELS_AVAILABLE:
            self.maxk_wrapper = MaxKSpmmWrapper()
        else:
            self.maxk_wrapper = None
        
        # Graph metadata (will be set during first forward pass)
        self.graph_indices = None
        self.graph_values = None
        self.graph_indptr = None
        self.node_degrees = None
        self.metadata_loaded = False
        self.use_maxk_kernel = False
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters exactly like DGL"""
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def set_graph_data(self, graph, graph_name=""):
        """
        Set graph data for MaxK kernel usage
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
        
        # Compute and store node degrees for proper normalization
        self.node_degrees = graph.in_degrees().float().to(indices.device)
        # Avoid division by zero for isolated nodes
        self.node_degrees = torch.clamp(self.node_degrees, min=1.0)
        
        print(f"📊 Graph stats: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
        print(f"📊 Degree stats: min={self.node_degrees.min():.1f}, max={self.node_degrees.max():.1f}, avg={self.node_degrees.mean():.1f}")
        
        # Load MaxK metadata if kernels are available
        if (MAXK_KERNELS_AVAILABLE and 
            graph_name and 
            self.maxk_wrapper and 
            self._aggre_type == "mean"):  # Only use MaxK for mean aggregation
            
            self.metadata_loaded = self.maxk_wrapper.load_metadata(graph_name)
            if self.metadata_loaded:
                self.use_maxk_kernel = True
                print(f"✅ MaxK metadata loaded for {graph_name}")
            else:
                print(f"⚠️ MaxK metadata failed, using DGL fallback for {graph_name}")
        else:
            print("⚠️ Using DGL fallback (no MaxK kernels or unsupported aggregator)")
            self.use_maxk_kernel = False
    
    def _lstm_reducer(self, nodes):
        """LSTM reducer (exactly like DGL)"""
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}
    
    def forward(self, graph, feat, edge_weight=None):
        """
        COMPLETE DGL-equivalent forward pass with MaxK acceleration
        Exactly replicates DGL SAGEConv.forward() behavior line-by-line
        """
        with graph.local_scope():
            # === STEP 1: Feature Processing (exactly like DGL) ===
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            
            # === STEP 2: Message Function Setup (exactly like DGL) ===
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")
            
            h_self = feat_dst  # Store raw features for self-connection (like DGL!)
            
            # === STEP 3: Handle Empty Graphs (exactly like DGL) ===
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)
                h_neigh = graph.dstdata["neigh"]
            else:
                # === STEP 4: Critical lin_before_mp Decision (exactly like DGL) ===
                lin_before_mp = self._in_src_feats > self._out_feats
                
                # === STEP 5: Aggregation Type Handling ===
                if self._aggre_type == "mean":
                    # Try MaxK acceleration if available, otherwise fall back to DGL
                    if (self.use_maxk_kernel and 
                        self.maxk_wrapper and 
                        self.node_degrees is not None and
                        edge_weight is None):  # MaxK doesn't support edge weights yet
                        
                        try:
                            # MaxK-accelerated mean aggregation
                            if lin_before_mp:
                                # Transform BEFORE aggregation (like DGL)
                                feat_to_aggregate = self.fc_neigh(feat_src)
                                h_neigh_sum = self.maxk_wrapper.spmm(
                                    self.graph_indices,
                                    self.graph_values,
                                    feat_to_aggregate,
                                    self.k_value,
                                    self.graph_indptr
                                )
                                # Apply mean normalization (convert sum to mean)
                                h_neigh = h_neigh_sum / self.node_degrees.unsqueeze(-1)
                            else:
                                # Aggregate THEN transform (like DGL)
                                h_neigh_sum = self.maxk_wrapper.spmm(
                                    self.graph_indices,
                                    self.graph_values,
                                    feat_src,  # Raw features
                                    self.k_value,
                                    self.graph_indptr
                                )
                                # Apply mean normalization first
                                h_neigh_mean = h_neigh_sum / self.node_degrees.unsqueeze(-1)
                                # Then transform
                                h_neigh = self.fc_neigh(h_neigh_mean)
                            
                            print(f"🚀 Used MaxK kernel for mean aggregation (lin_before_mp={lin_before_mp})")
                            
                        except Exception as e:
                            print(f"⚠️ MaxK kernel failed: {e}, falling back to DGL")
                            # Fall back to DGL implementation
                            graph.srcdata["h"] = (
                                self.fc_neigh(feat_src) if lin_before_mp else feat_src
                            )
                            graph.update_all(msg_fn, fn.mean("m", "neigh"))
                            h_neigh = graph.dstdata["neigh"]
                            if not lin_before_mp:
                                h_neigh = self.fc_neigh(h_neigh)
                    else:
                        # Standard DGL mean aggregation
                        graph.srcdata["h"] = (
                            self.fc_neigh(feat_src) if lin_before_mp else feat_src
                        )
                        graph.update_all(msg_fn, fn.mean("m", "neigh"))
                        h_neigh = graph.dstdata["neigh"]
                        if not lin_before_mp:
                            h_neigh = self.fc_neigh(h_neigh)
                
                elif self._aggre_type == "gcn":
                    # GCN aggregation (exactly like DGL)
                    check_eq_shape(feat)
                    graph.srcdata["h"] = (
                        self.fc_neigh(feat_src) if lin_before_mp else feat_src
                    )
                    if isinstance(feat, tuple):  # heterogeneous
                        graph.dstdata["h"] = (
                            self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                        )
                    else:
                        if graph.is_block:
                            graph.dstdata["h"] = graph.srcdata["h"][:graph.num_dst_nodes()]
                        else:
                            graph.dstdata["h"] = graph.srcdata["h"]
                    graph.update_all(msg_fn, fn.sum("m", "neigh"))
                    # Divide by in_degrees (GCN normalization)
                    degs = graph.in_degrees().to(feat_dst)
                    h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                        degs.unsqueeze(-1) + 1
                    )
                    if not lin_before_mp:
                        h_neigh = self.fc_neigh(h_neigh)
                
                elif self._aggre_type == "pool":
                    # Pool aggregation (exactly like DGL)
                    graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                    graph.update_all(msg_fn, fn.max("m", "neigh"))
                    h_neigh = self.fc_neigh(graph.dstdata["neigh"])
                
                elif self._aggre_type == "lstm":
                    # LSTM aggregation (exactly like DGL)
                    graph.srcdata["h"] = feat_src
                    graph.update_all(msg_fn, self._lstm_reducer)
                    h_neigh = self.fc_neigh(graph.dstdata["neigh"])
                
                else:
                    raise KeyError(f"Aggregator type {self._aggre_type} not recognized.")
            
            # === STEP 6: Combine Self and Neighbor Features (exactly like DGL) ===
            if self._aggre_type == "gcn":
                rst = h_neigh
                # Add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh  # Transform self connection here!
            
            # === STEP 7: Post-processing (exactly like DGL) ===
            if self.activation is not None:
                rst = self.activation(rst)
            if self.norm is not None:
                rst = self.norm(rst)
            
            return rst
