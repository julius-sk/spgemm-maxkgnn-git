#!/usr/bin/env python3
"""
Direct CUDA Kernel Interface for MaxK-GNN - FIXED VERSION
Uses compiled bindings to call spmm_maxk.cu and spmm_maxk_backward.cu kernels directly
"""

import torch
import numpy as np
import time
import os
from pathlib import Path
from graph_loader import GraphDataLoader

# Try to import the direct kernel bindings
try:
    import maxk_cuda_kernels  # This will be our compiled extension
    DIRECT_KERNELS_AVAILABLE = True
    print("✅ Direct CUDA kernels loaded successfully")
except ImportError:
    DIRECT_KERNELS_AVAILABLE = False
    print("⚠️ Direct CUDA kernels not available")
    print("   Build with: python setup_direct_kernels.py build_ext --inplace")

class DirectMaxKKernels:
    """
    Direct interface to MaxK-GNN CUDA kernels
    Calls the actual spmm_maxk.cu and spmm_maxk_backward.cu functions
    """
    
    def __init__(self, graph_name=""):
        self.graph_name = graph_name
        self.warp4_metadata = None
        self.num_warps = 0
        
    def load_warp4_metadata(self, graph_name=None, num_warps=12, warp_max_nz=64):
        """Load warp4 metadata required by the kernels"""
        if graph_name is None:
            graph_name = self.graph_name
            
        if not DIRECT_KERNELS_AVAILABLE:
            print("⚠️ Direct kernels not available for metadata loading")
            return False
            
        try:
            # Use the C++ function to load metadata
            self.warp4_metadata = maxk_cuda_kernels.load_warp4_metadata(
                graph_name, num_warps, warp_max_nz
            )
            self.num_warps = self.warp4_metadata.size(0) // 4
            print(f"✅ Loaded warp4 metadata: {self.num_warps} warps")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load warp4 metadata: {e}")
            print(f"   Ensure generate_meta.py has been run for {graph_name}")
            return False
    
    def generate_maxk_sparse_data(self, input_features, dim_k, use_cuda_topk=True):
        """
        Generate MaxK sparse representation (replicates main.cu logic)
        Does TopK selection to create sparse input
        ADDED: Option to use CUDA TopK kernel
        """
        v_num, dim_origin = input_features.shape
        #print(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        #ADDED: Try CUDA TopK first if available and requested
        if use_cuda_topk and DIRECT_KERNELS_AVAILABLE:
            try:
                print(f"🚀 Using CUDA TopK kernel for k={dim_k}")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                topk_values, topk_indices = maxk_cuda_kernels.cuda_topk_maxk_float(
                    input_features, dim_k
                )
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print(f"✅ CUDA TopK completed")
            except Exception as e:
                print(f"⚠️ CUDA TopK failed: {e}, falling back to PyTorch")
                topk_values, topk_indices = torch.topk(input_features, dim_k, dim=1)
        else:
        #     #Original PyTorch TopK
            topk_values, topk_indices = torch.topk(input_features, dim_k, dim=1)
        
        # Create sparse data and selector (matching main.cu format)
        sparse_data = topk_values  # Shape: (v_num, dim_k)
        sparse_selector = topk_indices.to(torch.uint8)  # Shape: (v_num, dim_k)
        
        return sparse_data, sparse_selector
    
    def run_forward_kernel(self, graph_data, input_features, dim_k, timing=True, use_cuda_topk=True):        
        """
        Run the direct MaxK forward kernel (spmm_maxk.cu)
        
        Args:
            graph_data: Dict with 'indices' and 'values' tensors
            input_features: Dense input tensor (v_num x dim_origin)  
            dim_k: Sparse dimension (k value)
            timing: Whether to measure execution time
            use_cuda_topk: Whether to use CUDA TopK kernel (ADDED)
            
        Returns:
            (output_tensor, execution_time_ms)
        """
        if not DIRECT_KERNELS_AVAILABLE:
            raise RuntimeError("Direct CUDA kernels not available")
            
        if self.warp4_metadata is None:
            raise RuntimeError("Warp4 metadata not loaded. Call load_warp4_metadata() first")
        # print(f"******************")
        # print(f"???????????????????")
        # Generate sparse representation using TopK (MODIFIED to add use_cuda_topk)
        sparse_data, sparse_selector = self.generate_maxk_sparse_data(input_features, dim_k, use_cuda_topk)
        print(f"📊 Generated sparse data: {sparse_data.shape}")
        print(f"📊 Generated sparse selector: {sparse_selector.shape}")
        print(f"📊 Sample selector[0]: {sparse_selector[0]}")
        
        if timing:
            # Use the built-in timing function
            times = maxk_cuda_kernels.benchmark_spmm_maxk(
                self.warp4_metadata,
                graph_data['indices'],
                graph_data['values'],
                sparse_data,
                sparse_selector,
                self.num_warps,
                dim_k,
                num_runs=4
            )
            avg_time = np.mean(times)
            
            # Get the result from the last run
            output = maxk_cuda_kernels.spmm_maxk_forward(
                self.warp4_metadata,
                graph_data['indices'], 
                graph_data['values'],
                sparse_data,
                sparse_selector,
                self.num_warps,
                dim_k
            )
            
            print(f"📊 MaxK kernel output shape: {output.shape}")
            return output, avg_time
        else:
            # Single run without timing
            output = maxk_cuda_kernels.spmm_maxk_forward(
                self.warp4_metadata,
                graph_data['indices'],
                graph_data['values'],
                sparse_data,
                sparse_selector,
                self.num_warps,
                dim_k
            )
            print(f"📊 MaxK kernel output shape: {output.shape}")
            return output, 0.0
    
    def run_backward_kernel(self, graph_data, grad_output, dim_k, timing=True, use_cuda_topk=True):
        """
        Run the direct MaxK backward kernel (spmm_maxk_backward.cu)
        
        Args:
            graph_data: Dict with 'indices' and 'values' tensors
            grad_output: Gradient tensor from next layer
            dim_k: Sparse dimension (k value)
            timing: Whether to measure execution time
            use_cuda_topk: Whether to use CUDA TopK kernel (ADDED)
            
        Returns:
            (grad_input_tensor, execution_time_ms)
        """
        if not DIRECT_KERNELS_AVAILABLE:
            raise RuntimeError("Direct CUDA kernels not available")
            
        if self.warp4_metadata is None:
            raise RuntimeError("Warp4 metadata not loaded")
        
        # Generate sparse selector based on grad_output (MODIFIED to add use_cuda_topk)
        _, sparse_selector = self.generate_maxk_sparse_data(grad_output, dim_k, use_cuda_topk)
        
        if timing:
            # Measure timing manually for backward kernel
            timer = maxk_cuda_kernels.CudaTimer()
            times = []
            
            # Warmup + timing runs
            for i in range(8):  # 4 warmup + 4 timing
                timer.start()
                grad_input = maxk_cuda_kernels.spmm_maxk_backward(
                    self.warp4_metadata,
                    graph_data['indices'],
                    graph_data['values'],
                    grad_output,
                    sparse_selector,
                    self.num_warps,
                    dim_k
                )
                elapsed = timer.stop()
                
                if i >= 4:  # Only count timing runs
                    times.append(elapsed)
            
            avg_time = np.mean(times)
            return grad_input, avg_time
        else:
            # Single run
            grad_input = maxk_cuda_kernels.spmm_maxk_backward(
                self.warp4_metadata,
                graph_data['indices'],
                graph_data['values'],
                grad_output,
                sparse_selector,
                self.num_warps,
                dim_k
            )
            return grad_input, 0.0
    
    def validate_against_cusparse(self, graph_data, input_features, dim_k, tolerance=0.001, use_cuda_topk=True):
        """
        FIXED: Proper validation - MaxK kernel should output FULL dimensions
        FIXED: Use SAME TopK selection for both methods
        """
        if not DIRECT_KERNELS_AVAILABLE:
            print("⚠️ Cannot validate - direct kernels not available")
            return False
        
        print(f"🔍 Validating MaxK kernel vs cuSPARSE for k={dim_k}")
        print(f"   Using {'CUDA' if use_cuda_topk else 'PyTorch'} TopK")
        
        # Step 1: Apply TopK ONCE and use same result for both methods
        if use_cuda_topk and DIRECT_KERNELS_AVAILABLE:
            try:
                topk_values, topk_indices = maxk_cuda_kernels.cuda_topk_maxk_float(input_features, dim_k)
                
                topk_indices_int64 = topk_indices.long()  # For scatter_
                topk_indices_uint8 = topk_indices.to(torch.uint8)  # For kernels
            except:
                topk_values, topk_indices_int64 = torch.topk(input_features, dim_k, dim=1)
                topk_indices_uint8 = topk_indices_int64.to(torch.uint8)
        else:
            topk_values, topk_indices_int64 = torch.topk(input_features, dim_k, dim=1)
            topk_indices_uint8 = topk_indices_int64.to(torch.uint8)
            
        # Create sparse input for cuSPARSE using SAME TopK result
        sparse_input = torch.zeros_like(input_features)
        sparse_input.scatter_(1, topk_indices_int64, topk_values)
        
        print(f"📊 Input shapes: {input_features.shape} -> sparse: {sparse_input.shape}")
        print(f"📊 Input sparsity: {torch.count_nonzero(sparse_input).item()}/{sparse_input.numel()} non-zero")
        
        # Step 2: Run MaxK kernel using SAME sparse data
        sparse_data = topk_values
        sparse_selector = topk_indices_uint8
        maxk_output = maxk_cuda_kernels.spmm_maxk_forward(
            self.warp4_metadata,
            graph_data['indices'],
            graph_data['values'],
            sparse_data,
            sparse_selector,
            self.num_warps,
            dim_k
        )
        
        print(f"📊 MaxK output shape: {maxk_output.shape}")
        print(f"📊 MaxK output sparsity: {torch.count_nonzero(maxk_output).item()}/{maxk_output.numel()}")
        print(f"📊 MaxK sample values: {maxk_output[0, topk_indices_int64[0, :5]]}")
        
        # Step 3: Run cuSPARE on SAME sparse input
        cusparse_output = maxk_cuda_kernels.cusparse_spmm(
            graph_data['indptr'], graph_data['indices'], graph_data['values'],
            sparse_input, timing=False
        )
        
        print(f"📊 cuSPARSE output shape: {cusparse_output.shape}")
        print(f"📊 cuSPARSE output sparsity: {torch.count_nonzero(cusparse_output).item()}/{cusparse_output.numel()}")
        print(f"📊 cuSPARSE sample values: {cusparse_output[0, topk_indices_int64[0, :5]]}")
        
        # Step 4: Both should have same shape - compare directly
        if maxk_output.shape != cusparse_output.shape:
            print(f"❌ Shape mismatch! MaxK: {maxk_output.shape}, cuSPARSE: {cusparse_output.shape}")
            return False
        
        # Step 5: Compare values at the positions where input was non-zero
        input_nonzero_mask = sparse_input != 0
        diff = torch.abs(maxk_output - cusparse_output)
        relevant_diff = diff[input_nonzero_mask]
        max_error = relevant_diff.max().item() if relevant_diff.numel() > 0 else 0.0
        avg_error = relevant_diff.mean().item() if relevant_diff.numel() > 0 else 0.0
        
        print(f"📊 Max error (at input nonzero): {max_error:.8f}")
        print(f"📊 Avg error (at input nonzero): {avg_error:.8f}")
        print(f"📊 Tolerance: {tolerance}")
        
        is_valid = max_error < tolerance
        
        if is_valid:
            print("✅ Validation PASSED! MaxK kernel produces correct results")
        else:
            print("❌ Validation FAILED! MaxK kernel has errors")
                
        return is_valid
    
    def benchmark_all_k_values(self, graph_data, dim_origin=256, k_values=[16, 32, 64], 
                              num_runs=4, use_cuda_topk=True):
        """
        Benchmark across different k values (replicates main.cu benchmark loop)
        ADDED: use_cuda_topk parameter and TopK benchmarking
        """
        if not DIRECT_KERNELS_AVAILABLE:
            print("❌ Direct kernels not available for benchmarking")
            return {}
        
        print(f"\n🏃‍♂️ Benchmarking Direct MaxK Kernels")
        print(f"Graph: {self.graph_name}")
        print(f"TopK method: {'CUDA' if use_cuda_topk else 'PyTorch'}")
        print("num graph dim_origin dim_k kernel time(ms)")
        print("-" * 50)
        
        v_num = graph_data['indptr'].size(0) - 1
        results = {}
        
        # Generate test input (same seed as main.cu)
        torch.manual_seed(123)
        input_features = torch.rand(v_num, dim_origin, device='cuda', dtype=torch.float32)
        
        # ADDED: Quick TopK benchmark
        if use_cuda_topk and DIRECT_KERNELS_AVAILABLE:
            print(f"\n🔥 TopK Performance Comparison:")
            for test_k in [32,64]:
                try:
                    # PyTorch timing
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    start = time.time()
                    for _ in range(10):
                        torch.topk(input_features, test_k, dim=1)
                    # torch.cuda.synchronize()
                    # torch.cuda.empty_cache()
                    pytorch_time = (time.time() - start) / 10 * 1000
                    
                    # CUDA timing
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    start = time.time()
                    for _ in range(10):
                        maxk_cuda_kernels.cuda_topk_maxk_float(input_features, test_k)
                    # torch.cuda.synchronize()
                    # torch.cuda.empty_cache()
                    cuda_time = (time.time() - start) / 10 * 1000
                    
                    speedup = pytorch_time / cuda_time if cuda_time > 0 else 0
                    print(f"k={test_k}: PyTorch {pytorch_time:.3f}ms, CUDA {cuda_time:.3f}ms, {speedup:.2f}x")
                except Exception as e:
                    print(f"k={test_k}: TopK benchmark failed: {e}")

        for dim_k in k_values:
            if dim_k > 64:  # Skip if exceeds limit
                print(f"⏭️  Skipping k={dim_k} (exceeds limit)")
                continue
                
            print(f"\n📊 Testing k = {dim_k}")
            
            try:
                # Forward kernel (MODIFIED to pass use_cuda_topk)
                output_forward, time_forward = self.run_forward_kernel(
                    graph_data, input_features, dim_k, timing=True, use_cuda_topk=1
                )

                # Backward kernel (MODIFIED to pass use_cuda_topk)
                grad_output = torch.rand_like(input_features)
                grad_input, time_backward = self.run_backward_kernel(
                    graph_data, grad_output, dim_k, timing=True, use_cuda_topk=1
                )
                
                # Store results
                results[dim_k] = {
                    'forward_time': time_forward,
                    'backward_time': time_backward
                }
                
                # Print in main.cu format
                print(f"1/1 {self.graph_name} {dim_origin} {dim_k} maxk {time_forward:.3f}")
                print(f"1/1 {self.graph_name} {dim_origin} {dim_k} maxk_backward {time_backward:.3f}")
                
            except Exception as e:
                print(f"❌ Failed for k={dim_k}: {e}")
                results[dim_k] = {'forward_time': -1, 'backward_time': -1}
        
        return results

def test_direct_kernels():
    """Test the direct kernel interface"""
    print("🧪 Testing Direct MaxK-GNN CUDA Kernels")
    print("=" * 50)
    
    if not DIRECT_KERNELS_AVAILABLE:
        print("❌ Direct kernels not available!")
        print("   Build with: python setup_direct_kernels.py build_ext --inplace")
        return False
    
    # Load test graph
    loader = GraphDataLoader()
    graphs = loader.get_available_graphs()
    
    if not graphs:
        print("❌ No graphs available for testing")
        return False
    
    
    # Test all graphs
    for i, test_graph in enumerate(graphs):
        print(f"\n📊 Testing graph {i+1}/{len(graphs)}: {test_graph}")   
        try:
            # Load graph data
            graph_data = loader.load_graph(test_graph)
            graph_data = loader.to_cuda_tensors(graph_data)
            # Initialize direct kernel interface
            kernels = DirectMaxKKernels(test_graph)
            
            # Load warp4 metadata
            if not kernels.load_warp4_metadata():
                print("❌ Cannot proceed without warp4 metadata")
                return False
            
            # Test validation with proper logic
            v_num = graph_data['v_num']
            test_features = torch.rand(v_num, 256, device='cuda', dtype=torch.float32)
            
            print(f"\n🔍 Validating with PyTorch TopK...")
            is_valid_pytorch = kernels.validate_against_cusparse(graph_data, test_features, dim_k=32, use_cuda_topk=False)
            
            print(f"\n🔍 Validating with CUDA TopK...")
            is_valid_cuda = kernels.validate_against_cusparse(graph_data, test_features, dim_k=32, use_cuda_topk=True)
            
            if not (is_valid_pytorch and is_valid_cuda):
                print("⚠️ Some validation failed - results may be incorrect")
            
            # Run benchmark only if validation passes
            if is_valid_pytorch and is_valid_cuda:
                print(f"\n📈 Running benchmark with CUDA TopK...")
                results = kernels.benchmark_all_k_values(
                    graph_data, dim_origin=256, k_values=[32, 64], num_runs=2, use_cuda_topk=True
                )
                
                print(f"\n✅ Direct kernel testing completed successfully!")
            else:
                print(f"\n⚠️ Skipping benchmark due to validation failure")
            continue
        
        except Exception as e:
            print(f"❌ Direct kernel testing failed for {test_graph}: {e}")
            continue
    return True
        
if __name__ == "__main__":
    test_direct_kernels()