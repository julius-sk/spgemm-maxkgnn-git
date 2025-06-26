#!/usr/bin/env python3
"""
Main Runner with Direct CUDA Kernel Bindings
Exact Python equivalent of main.cu using direct kernel calls
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import time

# Import our interfaces
from graph_loader import GraphDataLoader
from cusparse_kernel import CuSPARSEKernel
from direct_kernel_interface import DirectMaxKKernels, DIRECT_KERNELS_AVAILABLE

def benchmark_single_graph_direct(graph_name, loader):
    """
    Benchmark single graph using direct CUDA kernel calls
    Exact replica of test_graph() function from main.cu
    """
    print(f"\n{'='*60}")
    print(f"🔬 Direct CUDA Kernel Benchmark: {graph_name}")
    print(f"{'='*60}")
    
    # Load graph data (same as main.cu)
    try:
        graph_data = loader.load_graph(graph_name)
        graph_data = loader.to_cuda_tensors(graph_data)
    except Exception as e:
        print(f"❌ Failed to load graph {graph_name}: {e}")
        return None
    
    # Parameters from main.cu
    dim_origin = 256
    dim_k_list = [16, 32, 64, 96, 128, 192]
    dim_k_limit = 64
    
    v_num = graph_data['v_num']
    e_num = graph_data['e_num']
    
    print(f"📊 Graph Info: {v_num} vertices, {e_num} edges, avg_degree={e_num/v_num:.1f}")
    
    # Initialize kernels
    cusparse_kernel = CuSPARSEKernel()
    
    if DIRECT_KERNELS_AVAILABLE:
        maxk_kernels = DirectMaxKKernels(graph_name)
        
        # Load warp4 metadata (critical for kernel execution)
        if not maxk_kernels.load_warp4_metadata():
            print("⚠️ Failed to load warp4 metadata - MaxK kernels may fail")
            maxk_available = False
        else:
            maxk_available = True
    else:
        print("⚠️ Direct CUDA kernels not available")
        maxk_available = False
    
    # Generate test data (same random seed as main.cu)
    torch.manual_seed(123)
    vin_sparse = torch.rand(v_num, dim_origin, device='cuda', dtype=torch.float32)
    
    # Results storage
    results = {
        'graph_name': graph_name,
        'v_num': v_num,
        'e_num': e_num,
        'avg_degree': e_num / v_num,
        'cusparse_times': {},
        'maxk_forward_times': {},
        'maxk_backward_times': {}
    }
    
    print(f"\n📊 Performance Benchmark (Direct CUDA Kernels)")
    print("num graph dim_origin dim_k kernel time(ms)")
    print("-" * 55)
    
    # Benchmark loop (exactly like main.cu)
    for n, dim_k in enumerate(dim_k_list):
        if dim_k > dim_k_limit:
            break
            
        outstr = f"1/1 {graph_name} {dim_origin} {dim_k}"
        print(f"\n📈 Testing dim_k = {dim_k}")
        
        # cuSPARSE baseline (only for first iteration, like main.cu)
        if n == 0:
            print("🔄 Running cuSPARSE baseline...")
            cusparse_times = cusparse_kernel.benchmark_cusparse(
                graph_data, dim_list=[dim_origin], num_runs=10
            )
            cusparse_time = cusparse_times[dim_origin]
            results['cusparse_times'][dim_origin] = cusparse_time
            print(f"{outstr} cusparse {cusparse_time:.3f}")
        
        # MaxK kernels (if available)
        if maxk_available:
            try:
                # Validation (only for first k value, like main.cu)
                if n == 0:
                    print("🔍 Validating kernel correctness...")
                    is_valid = maxk_kernels.validate_against_cusparse(
                        graph_data, vin_sparse, dim_k, tolerance=0.001
                    )
                    if not is_valid:
                        print("❌ Validation failed - results may be incorrect")
                
                # Forward kernel timing
                print(f"⚡ Running MaxK forward kernel (k={dim_k})...")
                output_forward, time_forward = maxk_kernels.run_forward_kernel(
                    graph_data, vin_sparse, dim_k, timing=True
                )
                results['maxk_forward_times'][dim_k] = time_forward
                print(f"{outstr} maxk {time_forward:.3f}")
                
                # Backward kernel timing
                print(f"⚡ Running MaxK backward kernel (k={dim_k})...")
                grad_output = torch.rand_like(vin_sparse)
                grad_input, time_backward = maxk_kernels.run_backward_kernel(
                    graph_data, grad_output, dim_k, timing=True
                )
                results['maxk_backward_times'][dim_k] = time_backward
                print(f"{outstr} maxk_backward {time_backward:.3f}")
                
            except Exception as e:
                print(f"❌ MaxK kernel failed for k={dim_k}: {e}")
                print(f"{outstr} maxk FAILED")
                print(f"{outstr} maxk_backward FAILED")
        else:
            print(f"{outstr} maxk UNAVAILABLE")
            print(f"{outstr} maxk_backward UNAVAILABLE")
    
    return results

def analyze_speedups(all_results):
    """
    Analyze speedups exactly like the MaxK-GNN paper
    Focus on graphs with average degree > 50
    """
    print(f"\n{'='*60}")
    print(f"📊 SPEEDUP ANALYSIS (Direct CUDA Kernels)")
    print(f"{'='*60}")
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Filter high-degree graphs (avg_degree > 50, like paper)
    high_degree_graphs = [r for r in all_results if r['avg_degree'] > 50 and r['maxk_forward_times']]
    
    if not high_degree_graphs:
        print("⚠️ No high-degree graphs with MaxK results for analysis")
        return
    
    print(f"\n🔍 High-degree graphs (avg_degree > 50): {len(high_degree_graphs)}")
    
    # Calculate speedups for different k values
    k_values = [16, 32, 64]
    speedup_data = {k: [] for k in k_values}
    
    print("\nGraph | Avg Deg | Speedup vs cuSPARSE")
    print("------|---------|--------------------")
    print("      |         | k=16 | k=32 | k=64")
    print("------|---------|------|------|------")
    
    for result in high_degree_graphs:
        graph_name = result['graph_name']
        avg_degree = result['avg_degree']
        
        # Get cuSPARSE baseline
        cusparse_time = result['cusparse_times'].get(256, None)
        if cusparse_time is None:
            continue
        
        speedups = {}
        for k in k_values:
            maxk_time = result['maxk_forward_times'].get(k, None)
            if maxk_time is not None and maxk_time > 0:
                speedup = cusparse_time / maxk_time
                speedups[k] = speedup
                speedup_data[k].append(speedup)
            else:
                speedups[k] = 0.0
        
        print(f"{graph_name[:5]:5s} | {avg_degree:7.1f} | " +
              f"{speedups.get(16, 0.0):4.1f}x | " +
              f"{speedups.get(32, 0.0):4.1f}x | " +
              f"{speedups.get(64, 0.0):4.1f}x")
    
    # Calculate average speedups (like paper results)
    print(f"\n🏆 Average Speedups vs cuSPARSE:")
    for k in k_values:
        if speedup_data[k]:
            avg_speedup = np.mean(speedup_data[k])
            print(f"  k={k:2d}: {avg_speedup:.2f}x average speedup")
        else:
            print(f"  k={k:2d}: No data available")
    
    # Compare with paper results
    paper_speedups = {16: 6.93, 32: 5.39, 64: 2.55}  # From paper
    print(f"\n📚 Paper Results Comparison:")
    for k in k_values:
        if speedup_data[k]:
            our_speedup = np.mean(speedup_data[k])
            paper_speedup = paper_speedups.get(k, 0)
            if paper_speedup > 0:
                ratio = our_speedup / paper_speedup
                print(f"  k={k:2d}: Our {our_speedup:.2f}x vs Paper {paper_speedup:.2f}x (ratio: {ratio:.2f})")
            else:
                print(f"  k={k:2d}: Our {our_speedup:.2f}x (no paper reference)")

def main():
    """Main function - exact replica of main() from main.cu"""
    parser = argparse.ArgumentParser(description='MaxK-GNN Direct CUDA Kernel Benchmark')
    parser.add_argument('graph', nargs='?', help='Specific graph to benchmark')
    parser.add_argument('--base-dir', default='kernels/graphs/', help='Graph directory')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation')
    args = parser.parse_args()
    
    print("🚀 MaxK-GNN Direct CUDA Kernel Benchmark")
    print("=" * 50)
    print("Exact Python equivalent of main.cu with direct kernel calls")
    
    # System info
    print(f"\n🖥️ System Info:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")
    
    print(f"Direct kernels: {DIRECT_KERNELS_AVAILABLE}")
    
    if not DIRECT_KERNELS_AVAILABLE:
        print("\n❌ Direct CUDA kernels not available!")
        print("   Build with: python setup_direct_kernels.py build_ext --inplace")
        print("   Ensure warp4 metadata exists: python kernels/generate_meta.py")
        return 1
    
    # Load graphs
    loader = GraphDataLoader(args.base_dir)
    available_graphs = loader.get_available_graphs()
    
    if not available_graphs:
        print(f"❌ No graphs found in {args.base_dir}")
        return 1
    
    print(f"\n📊 Available graphs: {len(available_graphs)}")
    for graph in available_graphs[:5]:
        print(f"  - {graph}")
    if len(available_graphs) > 5:
        print(f"  ... and {len(available_graphs) - 5} more")
    
    # Benchmark
    all_results = []
    
    if args.graph:
        # Single graph mode (like ./main.cu reddit.dgl)
        if args.graph in available_graphs:
            result = benchmark_single_graph_direct(args.graph, loader)
            if result:
                all_results.append(result)
        else:
            print(f"❌ Graph '{args.graph}' not found!")
            return 1
    else:
        # All graphs mode (like ./main.cu)
        print(f"\n🔄 Benchmarking all {len(available_graphs)} graphs...")
        
        for i, graph_name in enumerate(available_graphs):
            print(f"\n[{i+1}/{len(available_graphs)}] Processing {graph_name}")
            
            try:
                result = benchmark_single_graph_direct(graph_name, loader)
                if result:
                    all_results.append(result)
                
                # GPU synchronization (like main.cu)
                torch.cuda.synchronize()
                
            except KeyboardInterrupt:
                print("\n⏹️ Benchmark interrupted")
                break
            except Exception as e:
                print(f"❌ Failed to benchmark {graph_name}: {e}")
                continue
    
    # Analysis
    if not args.validate_only:
        analyze_speedups(all_results)
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")
    
    print(f"\n✅ Direct kernel benchmark completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())