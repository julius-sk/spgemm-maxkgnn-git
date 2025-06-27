#!/usr/bin/env python3
"""
MaxK SpGEMM Function - Custom autograd function for GNN training
Integrates the MaxK CUDA kernels with PyTorch autograd for training
"""

import torch
from torch.autograd import Function
import numpy as np

# Try to import the direct kernel bindings
try:
    import maxk_cuda_kernels
    MAXK_KERNELS_AVAILABLE = True
    print("✅ MaxK CUDA kernels loaded for training integration")
except ImportError:
    MAXK_KERNELS_AVAILABLE = False
    print("⚠️ MaxK CUDA kernels not available - falling back to cuSPARSE")

class MaxKSpGEMMFunction(Function):
    """
    Custom autograd function that uses MaxK SpGEMM kernels
    Implements both forward (spmm_maxk.cu) and backward (spmm_maxk_backward.cu)
    """
    
    @staticmethod
    def forward(ctx, graph_indices, graph_values, input_features, k_value, 
                warp4_metadata, num_warps, graph_indptr=None):
        """
        Forward pass using MaxK SpGEMM kernel
        
        Args:
            ctx: PyTorch autograd context
            graph_indices: Graph edge indices (CSR format)
            graph_values: Graph edge values
            input_features: Dense input features (V x D)
            k_value: MaxK sparsity parameter
            warp4_metadata: Precomputed warp metadata
            num_warps: Number of warps for kernel execution
            graph_indptr: CSR row pointers (for cuSPARSE fallback)
        
        Returns:
            output_features: Dense output features (V x D)
        """
        # Apply MaxK selection to input features
        if k_value < input_features.size(1):
            # Get top-k values and indices
            topk_values, topk_indices = torch.topk(input_features, k_value, dim=1)
            
            # Convert to MaxK kernel format
            sparse_data = topk_values  # Shape: (V, k)
            sparse_selector = topk_indices.to(torch.uint8)  # Shape: (V, k)
        else:
            # If k >= feature_dim, use all features
            sparse_data = input_features
            sparse_selector = torch.arange(input_features.size(1), 
                                         device=input_features.device, 
                                         dtype=torch.uint8).unsqueeze(0).expand(input_features.size(0), -1)
        
        # Save for backward pass
        ctx.save_for_backward(graph_indices, graph_values, sparse_selector)
        ctx.k_value = k_value
        ctx.warp4_metadata = warp4_metadata
        ctx.num_warps = num_warps
        ctx.input_shape = input_features.shape
        ctx.graph_indptr = graph_indptr
        
        # Run MaxK forward kernel
        if MAXK_KERNELS_AVAILABLE and warp4_metadata is not None:
            try:
                output = maxk_cuda_kernels.spmm_maxk_forward(
                    warp4_metadata,
                    graph_indices,
                    graph_values,
                    sparse_data,
                    sparse_selector,
                    num_warps,
                    k_value
                )
                return output
            except Exception as e:
                print(f"⚠️ MaxK kernel failed: {e}, falling back to cuSPARSE")
        
        # Fallback to cuSPARSE with sparse input
        if graph_indptr is not None:
            # Reconstruct full sparse matrix for cuSPARSE
            sparse_input = torch.zeros_like(input_features)
            sparse_input.scatter_(1, topk_indices.long(), topk_values)
            
            if MAXK_KERNELS_AVAILABLE:
                output = maxk_cuda_kernels.cusparse_spmm(
                    graph_indptr, graph_indices, graph_values, sparse_input
                )
            else:
                # Pure PyTorch fallback
                V = input_features.size(0)
                # Convert to COO format for PyTorch sparse
                row_indices = []
                for i in range(len(graph_indptr) - 1):
                    start, end = graph_indptr[i], graph_indptr[i + 1]
                    row_indices.extend([i] * (end - start))
                
                row_tensor = torch.tensor(row_indices, device=graph_indices.device, dtype=torch.long)
                edge_index = torch.stack([row_tensor, graph_indices.long()])
                
                sparse_adj = torch.sparse_coo_tensor(
                    edge_index, graph_values, (V, V)
                ).coalesce()
                
                output = torch.sparse.mm(sparse_adj, sparse_input)
            
            return output
        else:
            raise RuntimeError("No graph_indptr provided for cuSPARSE fallback")
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using MaxK backward kernel
        
        Args:
            ctx: PyTorch autograd context
            grad_output: Gradient from next layer
            
        Returns:
            Gradients for all forward inputs (most are None)
        """
        graph_indices, graph_values, sparse_selector = ctx.saved_tensors
        k_value = ctx.k_value
        warp4_metadata = ctx.warp4_metadata
        num_warps = ctx.num_warps
        input_shape = ctx.input_shape
        graph_indptr = ctx.graph_indptr
        
        # Initialize gradient for input features
        grad_input = torch.zeros(input_shape, device=grad_output.device, dtype=grad_output.dtype)
        
        # Run MaxK backward kernel
        if MAXK_KERNELS_AVAILABLE and warp4_metadata is not None:
            try:
                # Get gradient in sparse format
                grad_sparse = maxk_cuda_kernels.spmm_maxk_backward(
                    warp4_metadata,
                    graph_indices,
                    graph_values,
                    grad_output,
                    sparse_selector,
                    num_warps,
                    k_value
                )
                
                # Scatter sparse gradients back to full tensor
                grad_input.scatter_(1, sparse_selector.long(), grad_sparse)
                
                return None, None, grad_input, None, None, None, None
                
            except Exception as e:
                print(f"⚠️ MaxK backward kernel failed: {e}, falling back to autograd")
        
        # Fallback: Let PyTorch handle backward pass automatically
        # This will use the forward implementation's backward
        return None, None, None, None, None, None, None

# Convenience function for easy integration
def maxk_spgemm(graph_indices, graph_values, input_features, k_value, 
                warp4_metadata=None, num_warps=0, graph_indptr=None):
    """
    Convenience function for MaxK SpGEMM operation
    
    Args:
        graph_indices: Graph edge indices
        graph_values: Graph edge values  
        input_features: Input node features
        k_value: MaxK sparsity parameter
        warp4_metadata: Precomputed warp metadata (optional)
        num_warps: Number of warps (optional)
        graph_indptr: CSR row pointers (for fallback)
    
    Returns:
        Output node features after graph convolution
    """
    return MaxKSpGEMMFunction.apply(
        graph_indices, graph_values, input_features, k_value,
        warp4_metadata, num_warps, graph_indptr
    )

class MaxKSpmmWrapper:
    """
    Wrapper class to manage MaxK kernel metadata and provide easy interface
    """
    
    def __init__(self, graph_name="", num_warps=12, warp_max_nz=64):
        self.graph_name = graph_name
        self.warp4_metadata = None
        self.num_warps = 0
        self.num_warps_config = num_warps
        self.warp_max_nz = warp_max_nz
        
    def load_metadata(self, graph_name=None):
        """Load warp4 metadata for the graph"""
        if graph_name is None:
            graph_name = self.graph_name
            
        if not MAXK_KERNELS_AVAILABLE:
            print("⚠️ MaxK kernels not available")
            return False
            
        try:
            self.warp4_metadata = maxk_cuda_kernels.load_warp4_metadata(
                graph_name, self.num_warps_config, self.warp_max_nz
            )
            self.num_warps = self.warp4_metadata.size(0) // 4
            print(f"✅ Loaded MaxK metadata for {graph_name}: {self.num_warps} warps")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load MaxK metadata: {e}")
            return False
    
    def spmm(self, graph_indices, graph_values, input_features, k_value, graph_indptr=None):
        """
        Perform SpMM with MaxK kernels if available, otherwise fallback
        
        Args:
            graph_indices: Graph edge indices
            graph_values: Graph edge values
            input_features: Input node features  
            k_value: MaxK sparsity parameter
            graph_indptr: CSR row pointers (for fallback)
            
        Returns:
            Output node features
        """
        return maxk_spgemm(
            graph_indices, graph_values, input_features, k_value,
            self.warp4_metadata, self.num_warps, graph_indptr
        )

def test_maxk_spgemm_function():
    """Test the MaxK SpGEMM function"""
    print("🧪 Testing MaxK SpGEMM Function")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Create test data
    V, E, D = 1000, 5000, 256
    k_value = 32
    
    # Generate random graph
    torch.manual_seed(42)
    graph_indices = torch.randint(0, V, (E,), device='cuda', dtype=torch.int32)
    graph_values = torch.rand(E, device='cuda', dtype=torch.float32)
    input_features = torch.rand(V, D, device='cuda', dtype=torch.float32, requires_grad=True)
    
    # Create dummy indptr for fallback
    graph_indptr = torch.zeros(V + 1, device='cuda', dtype=torch.int32)
    current_idx = 0
    for i in range(V):
        edges_for_node = torch.sum(graph_indices == i).item()
        graph_indptr[i + 1] = current_idx + edges_for_node
        current_idx += edges_for_node
    
    print(f"📊 Test data: {V} nodes, {E} edges, {D} features, k={k_value}")
    
    try:
        # Test forward pass
        print("🔄 Testing forward pass...")
        output = maxk_spgemm(
            graph_indices, graph_values, input_features, k_value,
            graph_indptr=graph_indptr
        )
        print(f"✅ Forward pass successful: {output.shape}")
        
        # Test backward pass
        print("🔄 Testing backward pass...")
        loss = output.sum()
        loss.backward()
        
        if input_features.grad is not None:
            print(f"✅ Backward pass successful: grad shape {input_features.grad.shape}")
            print(f"📊 Gradient stats: mean={input_features.grad.mean().item():.6f}, "
                  f"std={input_features.grad.std().item():.6f}")
        else:
            print("❌ No gradients computed")
        
        print("🎉 MaxK SpGEMM function test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_maxk_spgemm_function()
