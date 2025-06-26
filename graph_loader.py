#!/usr/bin/env python3
"""
Graph Data Loader - Python equivalent of the data loading functionality in main.cu
Loads graph data from .indptr and .indices files similar to cuda_read_array function
"""

import numpy as np
import torch
import os
from pathlib import Path
import struct

class GraphDataLoader:
    """Loads graph data in the same format as main.cu"""
    
    def __init__(self, base_dir="kernels/graphs/"):
        self.base_dir = base_dir
        
    def read_binary_array(self, filepath, dtype=np.int32):
        """
        Python equivalent of cuda_read_array template function
        Reads binary array from file and returns numpy array
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Graph file not found: {filepath}")
            
        # Read binary data
        with open(filepath, 'rb') as f:
            data = f.read()
        
        # Convert to numpy array
        if dtype == np.int32:
            arr = np.frombuffer(data, dtype=np.int32)
        elif dtype == np.float32:
            arr = np.frombuffer(data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        return arr
    
    def load_graph(self, graph_name):
        """
        Load graph data similar to main.cu test_graph function
        
        Args:
            graph_name: Name of the graph (e.g., "reddit.dgl")
            
        Returns:
            dict with indptr, indices, values, v_num, e_num
        """
        # Remove extension if present
        graph_stem = Path(graph_name).stem
        
        # Load indptr and indices files
        indptr_file = os.path.join(self.base_dir, f"{graph_stem}.indptr")
        indices_file = os.path.join(self.base_dir, f"{graph_stem}.indices")
        
        print(f"Loading graph: {graph_stem}")
        print(f"  indptr file: {indptr_file}")
        print(f"  indices file: {indices_file}")
        
        # Read arrays
        indptr = self.read_binary_array(indptr_file, np.int32)
        indices = self.read_binary_array(indices_file, np.int32)
        
        # Calculate graph statistics
        v_num = len(indptr) - 1  # Number of vertices
        e_num = len(indices)     # Number of edges
        
        # Generate values (similar to main.cu input_mode=1)
        np.random.seed(123)  # Same seed as main.cu
        values = np.random.uniform(0, 1, e_num).astype(np.float32)
        
        print(f"  Vertices: {v_num}")
        print(f"  Edges: {e_num}")
        print(f"  Average degree: {e_num/v_num:.2f}")
        
        return {
            'graph_name': graph_stem,
            'indptr': indptr,
            'indices': indices,
            'values': values,
            'v_num': v_num,
            'e_num': e_num
        }
    
    def to_cuda_tensors(self, graph_data, device='cuda'):
        """Convert numpy arrays to CUDA tensors"""
        cuda_data = {}
        for key, value in graph_data.items():
            if isinstance(value, np.ndarray):
                if value.dtype == np.int32:
                    cuda_data[key] = torch.from_numpy(value).int().to(device)
                elif value.dtype == np.float32:
                    cuda_data[key] = torch.from_numpy(value).float().to(device)
                else:
                    cuda_data[key] = torch.from_numpy(value).to(device)
            else:
                cuda_data[key] = value
        return cuda_data
    
    def generate_test_features(self, v_num, dim_origin=256, dim_k_limit=64, device='cuda'):
        """
        Generate test feature matrices similar to main.cu
        
        Returns:
            dict with sparse and dense feature matrices
        """
        # Set random seed like main.cu
        torch.manual_seed(123)
        
        # Generate dense input features (v_num x dim_origin)
        vin_sparse = torch.rand(v_num, dim_origin, device=device, dtype=torch.float32)
        
        # Generate sparse data for MaxK (v_num x dim_k_limit)
        vin_sparse_data = torch.rand(v_num, dim_k_limit, device=device, dtype=torch.float32)
        
        # Generate sparse selector (similar to main.cu sampling logic)
        vin_sparse_selector = torch.zeros(v_num, dim_k_limit, device=device, dtype=torch.uint8)
        
        # Sample indices for sparse representation (like main.cu)
        for i in range(v_num):
            # Sample without replacement
            indices = torch.randperm(dim_origin)[:dim_k_limit]
            vin_sparse_selector[i] = indices.to(torch.uint8)
        
        # Output tensors
        vout_ref = torch.zeros(v_num, dim_origin, device=device, dtype=torch.float32)
        vout_maxk = torch.zeros(v_num, dim_origin, device=device, dtype=torch.float32)
        vout_maxk_backward = torch.zeros(v_num, dim_k_limit, device=device, dtype=torch.float32)
        
        return {
            'vin_sparse': vin_sparse,
            'vin_sparse_data': vin_sparse_data, 
            'vin_sparse_selector': vin_sparse_selector,
            'vout_ref': vout_ref,
            'vout_maxk': vout_maxk,
            'vout_maxk_backward': vout_maxk_backward,
            'dim_origin': dim_origin,
            'dim_k_limit': dim_k_limit
        }
    
    def get_available_graphs(self):
        """Get list of available graph files"""
        if not os.path.exists(self.base_dir):
            return []
            
        graphs = []
        for file in os.listdir(self.base_dir):
            if file.endswith('.indptr'):
                graph_name = file[:-7]  # Remove .indptr extension
                # Check if corresponding .indices file exists
                indices_file = os.path.join(self.base_dir, f"{graph_name}.indices")
                if os.path.exists(indices_file):
                    graphs.append(graph_name)
        
        return sorted(graphs)

def test_graph_loader():
    """Test the graph loader functionality"""
    print("🧪 Testing Graph Data Loader")
    print("=" * 40)
    
    loader = GraphDataLoader()
    
    # Get available graphs
    graphs = loader.get_available_graphs()
    print(f"Available graphs: {graphs}")
    
    if not graphs:
        print("❌ No graphs found. Please ensure graph files are in ../graphs/")
        return
    
    # Test loading first available graph
    test_graph = graphs[0]
    print(f"\n📊 Testing with graph: {test_graph}")
    
    try:
        # Load graph data
        graph_data = loader.load_graph(test_graph)
        
        # Convert to CUDA tensors
        if torch.cuda.is_available():
            cuda_data = loader.to_cuda_tensors(graph_data)
            print(f"✅ Converted to CUDA tensors")
            
            # Generate test features
            features = loader.generate_test_features(
                cuda_data['v_num'], dim_origin=256, dim_k_limit=64
            )
            print(f"✅ Generated test features")
            print(f"   Dense features: {features['vin_sparse'].shape}")
            print(f"   Sparse data: {features['vin_sparse_data'].shape}")
            print(f"   Sparse selector: {features['vin_sparse_selector'].shape}")
            
        else:
            print("⚠️ CUDA not available, skipping tensor conversion")
            
        print(f"✅ Graph loading test successful!")
        
    except Exception as e:
        print(f"❌ Graph loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graph_loader()