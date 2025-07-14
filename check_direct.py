#!/usr/bin/env python3
"""
Reddit Graph Structure Checker
Comprehensive analysis to determine if Reddit graph is directed/undirected
and whether transpose operations are needed for backward pass
Uses the EXACT same loading method as your training code
"""

import torch
import dgl
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from dgl.data import RedditDataset
from dgl import AddSelfLoop
import time

def check_reddit_graph_structure(data_path="./data/"):
    """
    Comprehensive check of Reddit graph structure using your exact loading method
    """
    print("🔍 REDDIT GRAPH STRUCTURE ANALYSIS")
    print("=" * 60)
    
    try:
        # Load Reddit graph exactly like your code
        print("📂 Loading Reddit graph using your method...")
        from dgl.data import RedditDataset
        from dgl import AddSelfLoop
        
        transform = AddSelfLoop()
        data = RedditDataset(transform=transform, raw_dir=data_path)
        g = data[0]
        g = g.int()  # Convert to int like your code
        
        print(f"✅ Graph loaded successfully using RedditDataset")
        
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        print("💡 Make sure reddit dataset can be downloaded to your data_path")
        print(f"💡 Trying data_path: {data_path}")
        return None
    
    # Basic properties
    print(f"\n📊 BASIC PROPERTIES:")
    print(f"   Number of nodes: {g.num_nodes():,}")
    print(f"   Number of edges: {g.num_edges():,}")
    print(f"   Is directed: {g.is_directed()}")
    print(f"   Device: {g.device}")
    
    # Get edge information
    src, dst = g.edges()
    print(f"   Source nodes range: {src.min()} to {src.max()}")
    print(f"   Destination nodes range: {dst.min()} to {dst.max()}")
    
    # Check for self-loops
    self_loops = (src == dst).sum().item()
    print(f"   Self-loops: {self_loops:,}")
    
    # 🔍 SYMMETRY CHECK (Most Important!)
    print(f"\n🔍 GRAPH SYMMETRY ANALYSIS:")
    
    # Method 1: Sample-based symmetry check
    print("   Method 1: Sample-based symmetry check...")
    sample_size = min(10000, g.num_edges())
    sample_indices = torch.randperm(g.num_edges())[:sample_size]
    
    sample_src = src[sample_indices]
    sample_dst = dst[sample_indices]
    
    symmetric_count = 0
    for i in range(len(sample_src)):
        s, d = sample_src[i].item(), sample_dst[i].item()
        if s != d:  # Skip self-loops
            # Check if reverse edge exists
            reverse_exists = g.has_edges_between(d, s).item()
            if reverse_exists:
                symmetric_count += 1
    
    symmetry_ratio = symmetric_count / len(sample_src)
    print(f"   Sample symmetry ratio: {symmetry_ratio:.4f}")
    
    # Method 2: Adjacency matrix symmetry check (for smaller graphs or sample)
    print("   Method 2: Adjacency matrix symmetry check...")
    
    if g.num_nodes() <= 50000:  # Only for manageable sizes
        try:
            adj = g.adjacency_matrix(scipy_fmt='csr')
            adj_T = adj.T
            
            # Check if matrices are equal
            diff = adj - adj_T
            is_symmetric = diff.nnz == 0
            
            print(f"   Adjacency matrix is symmetric: {is_symmetric}")
            if not is_symmetric:
                print(f"   Number of asymmetric entries: {diff.nnz:,}")
                
        except Exception as e:
            print(f"   ⚠️ Adjacency matrix check failed: {e}")
    else:
        print("   ⚠️ Graph too large for full adjacency matrix check")
    
    # Method 3: Direct edge comparison
    print("   Method 3: Direct edge pair analysis...")
    
    # Create edge set for fast lookup
    edge_set = set(zip(src.cpu().numpy(), dst.cpu().numpy()))
    reverse_edge_set = set(zip(dst.cpu().numpy(), src.cpu().numpy()))
    
    # Find intersection
    symmetric_edges = edge_set.intersection(reverse_edge_set)
    
    print(f"   Total unique edges: {len(edge_set):,}")
    print(f"   Symmetric edge pairs: {len(symmetric_edges):,}")
    print(f"   Symmetry percentage: {len(symmetric_edges)/len(edge_set)*100:.2f}%")
    
    # 🔍 DEGREE ANALYSIS
    print(f"\n📈 DEGREE ANALYSIS:")
    
    in_degrees = g.in_degrees().float()
    out_degrees = g.out_degrees().float()
    
    print(f"   In-degree:  min={in_degrees.min():.0f}, max={in_degrees.max():.0f}, mean={in_degrees.mean():.2f}")
    print(f"   Out-degree: min={out_degrees.min():.0f}, max={out_degrees.max():.0f}, mean={out_degrees.mean():.2f}")
    
    # Check if in-degrees equal out-degrees (sign of undirected graph)
    degree_diff = (in_degrees - out_degrees).abs()
    nodes_with_equal_degrees = (degree_diff < 1e-6).sum().item()
    degree_equality_ratio = nodes_with_equal_degrees / g.num_nodes()
    
    print(f"   Nodes with equal in/out degrees: {nodes_with_equal_degrees:,} ({degree_equality_ratio:.2%})")
    
    # 🔍 TRANSPOSE REQUIREMENT ANALYSIS
    print(f"\n🎯 TRANSPOSE REQUIREMENT ANALYSIS:")
    
    # Determine if transpose is needed
    if symmetry_ratio > 0.95 and degree_equality_ratio > 0.95:
        transpose_needed = False
        graph_type = "UNDIRECTED"
    elif symmetry_ratio < 0.6:
        transpose_needed = True
        graph_type = "DIRECTED"
    else:
        transpose_needed = True  # Be safe
        graph_type = "MIXED/UNCLEAR"
    
    print(f"   Graph type: {graph_type}")
    print(f"   Transpose needed for backward pass: {transpose_needed}")
    
    if transpose_needed:
        print(f"   ⚠️  Your MaxKSpGEMMFunction.backward() NEEDS transpose!")
        print(f"   📝 Forward:  output = A @ input")
        print(f"   📝 Backward: grad_input = A.T @ grad_output")
    else:
        print(f"   ✅ Your current backward implementation should work!")
        print(f"   📝 Forward:  output = A @ input") 
        print(f"   📝 Backward: grad_input = A @ grad_output (same as forward)")
    
    # 🔍 CSR FORMAT ANALYSIS
    print(f"\n📋 CSR FORMAT ANALYSIS:")
    
    # Get CSR representation
    indptr, indices, edge_ids = g.adj_sparse('csr')
    
    print(f"   CSR indptr shape: {indptr.shape}")
    print(f"   CSR indices shape: {indices.shape}")
    print(f"   CSR indices range: {indices.min()} to {indices.max()}")
    print(f"   Average edges per node: {len(indices) / len(indptr):.2f}")
    
    # Check sorted property
    is_sorted = True
    for i in range(len(indptr) - 1):
        start, end = indptr[i], indptr[i + 1]
        if end > start:
            node_edges = indices[start:end]
            if not torch.all(node_edges[:-1] <= node_edges[1:]):
                is_sorted = False
                break
    
    print(f"   CSR indices are sorted: {is_sorted}")
    
    # 🔍 MEMORY AND PERFORMANCE IMPLICATIONS
    print(f"\n💾 MEMORY & PERFORMANCE IMPLICATIONS:")
    
    avg_degree = g.num_edges() / g.num_nodes()
    print(f"   Average degree: {avg_degree:.2f}")
    
    if transpose_needed:
        print(f"   📊 Need to store/compute transposed graph structure")
        print(f"   📊 Backward pass requires different adjacency matrix")
        print(f"   📊 May need separate CSC format or transpose operation")
    else:
        print(f"   📊 Can reuse same graph structure for forward/backward")
        print(f"   📊 No additional transpose operations needed")
        print(f"   📊 Memory efficient - single adjacency representation")
    
    # 🔍 RECOMMENDATIONS
    print(f"\n💡 RECOMMENDATIONS:")
    
    if transpose_needed:
        print(f"   1. ❌ Current MaxKSpGEMMFunction.backward() is INCORRECT")
        print(f"   2. 🔧 Need to implement graph transpose in backward pass")
        print(f"   3. 📝 Options:")
        print(f"      a) Pre-compute CSC format for transpose")
        print(f"      b) Add transpose flag to CUDA kernel")
        print(f"      c) Create separate transposed graph structure")
        print(f"   4. 🧪 This could explain poor training performance!")
    else:
        print(f"   1. ✅ Current MaxKSpGEMMFunction.backward() should be correct")
        print(f"   2. 🔍 Performance issues likely elsewhere:")
        print(f"      a) Double MaxK application")
        print(f"      b) Missing degree normalization") 
        print(f"      c) CBSR format issues")
        print(f"      d) Kernel implementation bugs")
    
    return {
        'is_directed': g.is_directed(),
        'symmetry_ratio': symmetry_ratio,
        'degree_equality_ratio': degree_equality_ratio,
        'transpose_needed': transpose_needed,
        'graph_type': graph_type,
        'num_nodes': g.num_nodes(),
        'num_edges': g.num_edges(),
        'avg_degree': avg_degree
    }

def create_test_graphs():
    """Create small test graphs to verify transpose logic"""
    print(f"\n🧪 CREATING TEST GRAPHS FOR VERIFICATION:")
    
    # Test 1: Clearly directed graph
    print(f"\n--- Test 1: Directed Graph ---")
    edges_directed = [(0, 1), (1, 2), (2, 0)]  # Triangle with one direction
    g_directed = dgl.graph(edges_directed, num_nodes=3)
    
    print(f"Directed test graph:")
    print(f"  Edges: {edges_directed}")
    print(f"  Is directed: {g_directed.is_directed()}")
    
    # Check symmetry
    src, dst = g_directed.edges()
    edge_set = set(zip(src.numpy(), dst.numpy()))
    reverse_set = set(zip(dst.numpy(), src.numpy()))
    symmetric_edges = edge_set.intersection(reverse_set)
    print(f"  Symmetric edges: {len(symmetric_edges)}/{len(edge_set)}")
    
    # Test 2: Clearly undirected graph  
    print(f"\n--- Test 2: Undirected Graph ---")
    edges_undirected = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
    g_undirected = dgl.graph(edges_undirected, num_nodes=3)
    
    print(f"Undirected test graph:")
    print(f"  Edges: {edges_undirected}")
    print(f"  Is directed: {g_undirected.is_directed()}")
    
    # Check symmetry
    src, dst = g_undirected.edges()
    edge_set = set(zip(src.numpy(), dst.numpy()))
    reverse_set = set(zip(dst.numpy(), src.numpy()))
    symmetric_edges = edge_set.intersection(reverse_set)
    print(f"  Symmetric edges: {len(symmetric_edges)}/{len(edge_set)}")

if __name__ == "__main__":
    # Run the analysis using your exact loading method
    print("🚀 Using your exact Reddit loading method:")
    print("   RedditDataset(transform=AddSelfLoop(), raw_dir=data_path)")
    print()
    
    result = check_reddit_graph_structure()
    
    # Create test graphs for verification
    create_test_graphs()
    
    print(f"\n" + "="*60)
    print(f"🎯 FINAL CONCLUSION:")
    
    if result:
        if result['transpose_needed']:
            print(f"❌ TRANSPOSE IS REQUIRED - Fix your backward pass!")
        else:
            print(f"✅ NO TRANSPOSE NEEDED - Look for other issues!")
            
        print(f"📊 Graph type: {result['graph_type']}")
        print(f"📊 Symmetry ratio: {result['symmetry_ratio']:.3f}")
        print(f"📊 Performance bottleneck likely: {'Transpose missing' if result['transpose_needed'] else 'Other issues (double MaxK, normalization, etc.)'}")
