#!/usr/bin/env python3
"""
Short test to confirm cuda_topk_maxk_float memory access bug for k < 19
"""

import torch

try:
    import maxk_cuda_kernels
    print("✅ Kernels loaded")
except ImportError:
    print("❌ Kernels not available")
    exit(1)

# Create test data
input_features = torch.rand(334925, 256, device='cuda', dtype=torch.float32)
print(f"📊 Input shape: {input_features.shape}")

# Test different k values around the boundary
test_k_values = [8, 16, 18, 19, 20, 32]

for k in test_k_values:
    try:
        print(f"\n🧪 Testing k={k}...")
        result_values, result_indices = maxk_cuda_kernels.cuda_topk_maxk_float(input_features, k)
        print(f"✅ k={k} SUCCESS - output shapes: {result_values.shape}, {result_indices.shape}")
    except Exception as e:
        print(f"❌ k={k} FAILED: {e}")
        if "illegal memory access" in str(e).lower():
            print(f"   🚨 CONFIRMED: Illegal memory access at k={k}")