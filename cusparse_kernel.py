#!/usr/bin/env python3
"""
cuSPARSE Kernel Python Interface
Python wrapper for cuSPARSE functionality, replicating spmm_cusparse.cu
"""

import torch
import numpy as np
import time
import os
from graph_loader import GraphDataLoader

# Try different cuSPARSE interfaces
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    CUPY_AVAILABLE = True
    print("✅ CuPy available for cuSPARSE operations")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy not available")

# Check if torch_sparse is available (alternative)
try:
    import torch_sparse
    TORCH_SPARSE_AVAILABLE = True
    print("✅ torch_sparse available")
except ImportError:
    TORCH_SPARSE_AVAILABLE = False
    print("⚠️ torch_sparse not available")

class CuSPARSEKernel:
    """Python interface for cuSPARSE SPMM operations"""
    
    def __init__(self):
        self.method = self._detect_best_method()
        print(f"🔧 Using method: {self.method}")
    
    def _detect_best_method(self):
        """Detect the best available method for sparse operations"""
        if CUPY_AVAILABLE:
            return "cupy"
        elif TORCH_SPARSE_AVAILABLE:
            return "torch_sparse"
        else:
            return "pytorch_native"
    
    def convert_csr_to_tensors(self, graph_data):
        """Convert graph data to CSR format tensors"""
        indptr = graph_data['indptr']
        indices = graph_data['indices']
        values = graph_data['values']
        v_num = graph_data['v_num']
        e_num = graph_data['e_num']
        
        return indptr, indices, values, v_num, e_num
    
    def spmm_cusparse_csr(self, graph_data, features, output, timing=True, num_warmup=4, num_runs=10):
        """
        Sparse Matrix-Matrix Multiplication using CSR format
        Replicates spmm_cusparse function from spmm_cusparse.cu
        
        Args:
            graph_data: Graph data with indptr, indices, values
            features: Dense input matrix (v_num x dim)
            output: Dense output matrix (v_num x dim) 
            timing: Whether to perform timing measurements
            num_warmup: Number of warmup iterations
            num_runs: Number of timing runs
            
        Returns:
            Average execution time in seconds
        """
        indptr, indices, values, v_num, e_num = self.convert_csr_to_tensors(graph_data)
        
        if self.method == "cupy":
            return self._spmm_cupy_csr(indptr, indices, values, features, output, 
                                     timing, num_warmup, num_runs)
        elif self.method == "torch_sparse":
            return self._spmm_torch_sparse(indptr, indices, values, features, output,
                                         timing, num_warmup, num_runs)
        else:
            return self._spmm_pytorch_native(indptr, indices, values, features, output,
                                           timing, num_warmup, num_runs)
    
    def _spmm_cupy_csr(self, indptr, indices, values, features, output, 
                      timing, num_warmup, num_runs):
        """cuSPARSE implementation using CuPy"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        # Convert to CuPy arrays
        indptr_cp = cp.asarray(indptr.cpu().numpy())
        indices_cp = cp.asarray(indices.cpu().numpy()) 
        values_cp = cp.asarray(values.cpu().numpy())
        features_cp = cp.asarray(features.cpu().numpy())
        
        v_num = features.shape[0]
        
        # Create CSR matrix
        csr_matrix = cp_sparse.csr_matrix((values_cp, indices_cp, indptr_cp), 
                                         shape=(v_num, v_num))
        
        if not timing:
            # Single run without timing
            result_cp = csr_matrix @ features_cp
            result_torch = torch.from_numpy(cp.asnumpy(result_cp)).cuda()
            output.copy_(result_torch)
            return 0.0
        
        # Warmup runs
        for _ in range(num_warmup):
            result_cp = csr_matrix @ features_cp
        
        cp.cuda.Device().synchronize()
        
        # Timing runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            result_cp = csr_matrix @ features_cp
            cp.cuda.Device().synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Copy final result back
        result_torch = torch.from_numpy(cp.asnumpy(result_cp)).cuda()
        output.copy_(result_torch)
        
        avg_time = np.mean(times)
        return avg_time
    
    def _spmm_torch_sparse(self, indptr, indices, values, features, output,
                          timing, num_warmup, num_runs):
        """torch_sparse implementation"""
        if not TORCH_SPARSE_AVAILABLE:
            raise RuntimeError("torch_sparse not available")
        
        # Convert CSR to COO format for torch_sparse
        row_indices = []
        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i + 1]
            row_indices.extend([i] * (end - start))
        
        row_tensor = torch.tensor(row_indices, device=indices.device)
        edge_index = torch.stack([row_tensor, indices])
        
        if not timing:
            # Single run
            result = torch_sparse.spmm(edge_index, values, features.shape[0], 
                                     features.shape[0], features)
            output.copy_(result)
            return 0.0
        
        # Warmup
        for _ in range(num_warmup):
            result = torch_sparse.spmm(edge_index, values, features.shape[0],
                                     features.shape[0], features)
        
        torch.cuda.synchronize()
        
        # Timing runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            result = torch_sparse.spmm(edge_index, values, features.shape[0],
                                     features.shape[0], features)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
        
        output.copy_(result)
        avg_time = np.mean(times)
        return avg_time
    
    def _spmm_pytorch_native(self, indptr, indices, values, features, output,
                           timing, num_warmup, num_runs):
        """PyTorch native sparse implementation"""
        
        # Convert CSR to COO format
        row_indices = []
        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i + 1]
            row_indices.extend([i] * (end - start))
        
        row_tensor = torch.tensor(row_indices, device=indices.device, dtype=torch.long)
        edge_index = torch.stack([row_tensor, indices.long()])
        
        v_num = features.shape[0]
        sparse_adj = torch.sparse_coo_tensor(edge_index, values, (v_num, v_num)).coalesce()
        
        if not timing:
            # Single run
            result = torch.sparse.mm(sparse_adj, features)
            output.copy_(result)
            return 0.0
        
        # Warmup
        for _ in range(num_warmup):
            result = torch.sparse.mm(sparse_adj, features)
        
        torch.cuda.synchronize()
        
        # Timing runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            result = torch.sparse.mm(sparse_adj, features)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
        
        output.copy_(result)
        avg_time = np.mean(times)
        return avg_time
    
    def benchmark_cusparse(self, graph_data, dim_list=[256], num_runs=10):
        """
        Benchmark cuSPARSE performance across different dimensions
        Replicates the benchmarking from main.cu
        """
        print(f"\n🏃‍♂️ Benchmarking cuSPARSE SPMM")
        print("num graph dim kernel time(ms)")
        print("-" * 40)
        
        results = {}
        v_num = graph_data['v_num']
        graph_name = graph_data['graph_name']
        
        for dim in dim_list:
            print(f"\n📊 Testing with dim = {dim}")
            
            # Generate random input features
            torch.manual_seed(123)
            features = torch.rand(v_num, dim, device='cuda', dtype=torch.float32)
            output = torch.zeros(v_num, dim, device='cuda', dtype=torch.float32)
            
            # Run benchmark
            avg_time = self.spmm_cusparse_csr(
                graph_data, features, output, 
                timing=True, num_warmup=4, num_runs=num_runs
            )
            
            avg_time_ms = avg_time * 1000
            results[dim] = avg_time_ms
            
            # Print result in main.cu format
            print(f"1/1 {graph_name} {dim} cusparse {avg_time_ms:.3f}")
            
        return results
    
    def validate_correctness(self, graph_data, reference_output=None, dim=256):
        """
        Validate correctness of cuSPARSE implementation
        Optionally compare against reference output
        """
        print(f"\n🔍 Validating cuSPARSE correctness")
        
        v_num = graph_data['v_num']
        
        # Generate deterministic input
        torch.manual_seed(42)
        features = torch.rand(v_num, dim, device='cuda', dtype=torch.float32)
        output = torch.zeros(v_num, dim, device='cuda', dtype=torch.float32)
        
        # Run cuSPARSE
        self.spmm_cusparse_csr(graph_data, features, output, timing=False)
        
        if reference_output is not None:
            # Compare with reference
            diff = torch.abs(output - reference_output)
            max_diff = torch.max(diff).item()
            avg_diff = torch.mean(diff).item()
            
            print(f"Max difference: {max_diff:.6f}")
            print(f"Average difference: {avg_diff:.6f}")
            
            if avg_diff < 1e-4:
                print("✅ Validation passed!")
                return True
            else:
                print("❌ Validation failed!")
                return False
        else:
            print("✅ cuSPARSE execution completed")
            return True

def test_cusparse_kernel():
    """Test the cuSPARSE kernel interface"""
    print("🧪 Testing cuSPARSE Kernel Interface")
    print("=" * 40)
    
    # Load graph data
    loader = GraphDataLoader()
    graphs = loader.get_available_graphs()
    
    if not graphs:
        print("❌ No graphs found. Creating dummy data for testing...")
        # Create dummy graph data
        v_num, e_num = 1000, 5000
        
        # Create a simple random graph
        torch.manual_seed(123)
        row_indices = torch.randint(0, v_num, (e_num,))
        col_indices = torch.randint(0, v_num, (e_num,))
        
        # Convert to CSR format
        edge_list = torch.stack([row_indices, col_indices]).t()
        edge_list = torch.unique(edge_list, dim=0)  # Remove duplicates
        e_num = edge_list.shape[0]
        
        # Sort by row indices for CSR format
        sorted_edges = edge_list[edge_list[:, 0].argsort()]
        
        # Build indptr
        indptr = torch.zeros(v_num + 1, dtype=torch.int32)
        current_row = 0
        for i, (row, col) in enumerate(sorted_edges):
            while current_row <= row:
                indptr[current_row + 1] = i
                current_row += 1
        indptr[current_row:] = e_num
        
        indices = sorted_edges[:, 1].int()
        values = torch.ones(e_num, dtype=torch.float32)
        
        graph_data = {
            'graph_name': 'dummy',
            'indptr': indptr.cuda(),
            'indices': indices.cuda(),
            'values': values.cuda(),
            'v_num': v_num,
            'e_num': e_num
        }
    else:
        test_graph = graphs[0]
        print(f"📊 Using graph: {test_graph}")
        graph_data = loader.load_graph(test_graph)
        graph_data = loader.to_cuda_tensors(graph_data)
    
    # Initialize cuSPARSE kernel
    cusparse = CuSPARSEKernel()
    
    try:
        # Test correctness
        cusparse.validate_correctness(graph_data, dim=64)
        
        # Run benchmark
        results = cusparse.benchmark_cusparse(
            graph_data, 
            dim_list=[64, 128, 256],  # Reduced for testing
            num_runs=5  # Reduced for testing
        )
        
        print("\n📈 Benchmark Results:")
        for dim, time_ms in results.items():
            print(f"  dim={dim}: {time_ms:.3f} ms")
        
        print("\n✅ cuSPARSE testing completed successfully!")
        
    except Exception as e:
        print(f"❌ cuSPARSE testing failed: {e}")
        import traceback
        traceback.print_exc()

def compare_with_maxk_kernels():
    """Compare cuSPARSE with MaxK kernels (if available)"""
    print("\n🔀 Comparing cuSPARSE vs MaxK kernels")
    print("=" * 45)
    
    # This would integrate with the SPMM MaxK kernel for comparison
    # Similar to the comparison done in main.cu
    
    loader = GraphDataLoader()
    graphs = loader.get_available_graphs()
    
    if not graphs:
        print("❌ No graphs available for comparison")
        return
    
    test_graph = graphs[0]
    graph_data = loader.load_graph(test_graph)
    graph_data = loader.to_cuda_tensors(graph_data)
    
    # Initialize kernels
    cusparse = CuSPARSEKernel()
    
    # Test with different dimensions
    dims_to_test = [64, 128, 256]
    
    print("Graph | Dim | cuSPARSE (ms) | Speedup vs cuSPARSE")
    print("-" * 60)
    
    for dim in dims_to_test:
        # Test cuSPARSE
        cusparse_results = cusparse.benchmark_cusparse(
            graph_data, dim_list=[dim], num_runs=3
        )
        cusparse_time = cusparse_results[dim]
        
        print(f"{test_graph[:8]:8s} | {dim:3d} | {cusparse_time:8.3f} | {'1.00x':>15s}")
    
    print("\n💡 To see MaxK kernel comparison, ensure spmm_kernels is built")

if __name__ == "__main__":
    test_cusparse_kernel()
    print("\n" + "="*50)
    compare_with_maxk_kernels()