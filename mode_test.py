#!/usr/bin/env python3
"""
Side-by-Side Model Comparison with Identical Initialization
Compares Original SAGE, Hybrid MaxK-SAGE, and Fixed MaxK-SAGE with exact same weights
"""

import torch
import torch.nn as nn
import dgl
import numpy as np
import copy
from collections import OrderedDict

# Import the model classes
from maxk_sage_fixed import SAGE, HybridMaxKSAGE, MaxKSAGE

def set_deterministic_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_test_graph(num_nodes=1000, num_edges=5000, seed=42):
    """Create a deterministic test graph"""
    set_deterministic_seed(seed)
    
    # Create edges
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    
    # Remove self-loops and duplicates for cleaner test
    edge_list = torch.stack([src, dst], dim=0).t()
    edge_list = edge_list[edge_list[:, 0] != edge_list[:, 1]]  # Remove self-loops
    edge_list = torch.unique(edge_list, dim=0)  # Remove duplicates
    
    g = dgl.graph((edge_list[:, 0], edge_list[:, 1]), num_nodes=num_nodes)
    
    return g

def copy_model_weights(source_model, target_model):
    """Copy weights from source model to target model"""
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    
    # Map between different layer naming conventions
    weight_mapping = {}
    
    # Print available parameters for debugging
    print(f"Source model parameters: {list(source_state.keys())}")
    print(f"Target model parameters: {list(target_state.keys())}")
    
    # Create mapping between parameter names
    for target_key in target_state.keys():
        # Try to find corresponding source key
        source_key = None
        
        # Direct match first
        if target_key in source_state:
            source_key = target_key
        else:
            # Handle different naming conventions
            if 'layers.' in target_key:
                # Extract layer number and parameter type
                parts = target_key.split('.')
                layer_idx = parts[1]
                param_type = '.'.join(parts[2:])
                
                # Look for corresponding parameter in source
                potential_keys = [
                    f"layers.{layer_idx}.{param_type}",
                    f"layers.{layer_idx}.fc_neigh.{param_type}",
                    f"layers.{layer_idx}.fc_self.{param_type}",
                ]
                
                for pk in potential_keys:
                    if pk in source_state:
                        source_key = pk
                        break
        
        if source_key:
            weight_mapping[target_key] = source_key
        else:
            print(f"⚠️ No mapping found for {target_key}")
    
    # Copy weights
    copied_count = 0
    for target_key, source_key in weight_mapping.items():
        if source_state[source_key].shape == target_state[target_key].shape:
            target_state[target_key].copy_(source_state[source_key])
            copied_count += 1
        else:
            print(f"⚠️ Shape mismatch: {target_key} {target_state[target_key].shape} vs {source_key} {source_state[source_key].shape}")
    
    target_model.load_state_dict(target_state)
    print(f"✅ Copied {copied_count} parameters")
    
    return copied_count

def initialize_model_identically(model_class, *args, **kwargs):
    """Initialize a model with deterministic weights"""
    set_deterministic_seed(42)  # Reset seed before each model creation
    model = model_class(*args, **kwargs)
    
    # Apply same initialization to all models
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    return model

def compare_models_side_by_side():
    """Compare three models with identical initialization"""
    print("🔬 Side-by-Side Model Comparison with Identical Initialization")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Test configuration
    num_nodes = 1000
    num_edges = 5000
    feat_dim = 128
    hidden_dim = 64
    output_dim = 10
    maxk = 32
    
    # Create deterministic test data
    print("📊 Creating test data...")
    g = create_test_graph(num_nodes, num_edges, seed=42).to('cuda')
    
    set_deterministic_seed(42)
    features = torch.randn(num_nodes, feat_dim).cuda()
    
    print(f"   Graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
    print(f"   Features: {features.shape}")
    print(f"   Average degree: {g.num_edges() * 2 / g.num_nodes():.1f}")
    
    # Create models with identical initialization
    print("\n🏗️ Creating models with identical initialization...")
    
    # Create base model first
    set_deterministic_seed(42)
    model_original = initialize_model_identically(
        SAGE, feat_dim, hidden_dim, 2, output_dim, maxk, 
        feat_drop=0.0, norm=False, nonlinear="maxk"
    ).cuda()
    
    set_deterministic_seed(42)
    model_hybrid = initialize_model_identically(
        HybridMaxKSAGE, feat_dim, hidden_dim, 2, output_dim, maxk,
        feat_drop=0.0, norm=False, nonlinear="maxk"
    ).cuda()
    
    set_deterministic_seed(42)
    model_fixed = initialize_model_identically(
        MaxKSAGE, feat_dim, hidden_dim, 2, output_dim, maxk,
        feat_drop=0.0, norm=False, nonlinear="maxk", graph_name="test"
    ).cuda()
    
    print("✅ Models created")
    
    # Verify identical initialization by checking a few key parameters
    print("\n🔍 Verifying identical initialization...")
    
    orig_params = dict(model_original.named_parameters())
    hybrid_params = dict(model_hybrid.named_parameters())
    
    # Check lin_in and lin_out weights (these should be identical)
    for param_name in ['lin_in.weight', 'lin_out.weight']:
        if param_name in orig_params and param_name in hybrid_params:
            orig_weight = orig_params[param_name]
            hybrid_weight = hybrid_params[param_name]
            
            diff = torch.abs(orig_weight - hybrid_weight).max().item()
            print(f"   {param_name}: max_diff = {diff:.8f}")
            
            if diff < 1e-6:
                print(f"   ✅ {param_name} identical")
            else:
                print(f"   ⚠️ {param_name} differs by {diff}")
    
    # Run forward passes
    print("\n🚀 Running forward passes...")
    
    models = {
        "Original SAGE": model_original,
        "Hybrid MaxK-SAGE": model_hybrid, 
        "Fixed MaxK-SAGE": model_fixed
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📊 Testing {name}...")
        model.eval()  # Set to eval mode to ensure deterministic behavior
        
        try:
            with torch.no_grad():  # No gradients for pure comparison
                output = model(g, features)
            
            # Compute statistics
            output_min = output.min().item()
            output_max = output.max().item()
            output_mean = output.mean().item()
            output_std = output.std().item()
            output_norm = output.norm().item()
            
            # Check for NaN or Inf
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            
            results[name] = {
                'output': output.cpu(),
                'min': output_min,
                'max': output_max,
                'mean': output_mean,
                'std': output_std,
                'norm': output_norm,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'success': True
            }
            
            print(f"   ✅ Success")
            print(f"   📈 Range: [{output_min:.6f}, {output_max:.6f}]")
            print(f"   📊 Mean: {output_mean:.6f}, Std: {output_std:.6f}")
            print(f"   🔢 L2 Norm: {output_norm:.6f}")
            
            if has_nan or has_inf:
                print(f"   ⚠️ Contains NaN: {has_nan}, Inf: {has_inf}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Compare outputs
    print(f"\n🔍 Detailed Comparison:")
    print("=" * 70)
    
    successful_models = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_models) >= 2:
        model_names = list(successful_models.keys())
        
        # Compare each pair
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                result1, result2 = successful_models[name1], successful_models[name2]
                
                output1 = result1['output']
                output2 = result2['output']
                
                # Compute differences
                abs_diff = torch.abs(output1 - output2)
                max_diff = abs_diff.max().item()
                mean_diff = abs_diff.mean().item()
                
                # Relative difference
                rel_diff = abs_diff / (torch.abs(output1) + 1e-8)
                max_rel_diff = rel_diff.max().item()
                mean_rel_diff = rel_diff.mean().item()
                
                # Correlation
                corr = torch.corrcoef(torch.stack([output1.flatten(), output2.flatten()]))[0, 1].item()
                
                print(f"\n📊 {name1} vs {name2}:")
                print(f"   Max absolute diff: {max_diff:.8f}")
                print(f"   Mean absolute diff: {mean_diff:.8f}")
                print(f"   Max relative diff: {max_rel_diff:.8f}")
                print(f"   Mean relative diff: {mean_rel_diff:.8f}")
                print(f"   Correlation: {corr:.8f}")
                
                # Interpretation
                if max_diff < 1e-5:
                    print("   ✅ Outputs are nearly identical")
                elif max_diff < 1e-3:
                    print("   ✅ Outputs are very similar")
                elif max_diff < 0.1:
                    print("   ⚠️ Outputs have small differences")
                else:
                    print("   ❌ Outputs differ significantly")
    
    # Summary table
    print(f"\n📋 Summary Table:")
    print("=" * 70)
    print(f"{'Model':<20} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        if result['success']:
            print(f"{name:<20} {result['min']:<12.6f} {result['max']:<12.6f} "
                  f"{result['mean']:<12.6f} {result['std']:<12.6f}")
        else:
            print(f"{name:<20} {'FAILED':<48}")
    
    print("\n💡 Interpretation:")
    print("- If outputs are nearly identical: Models are equivalent")
    print("- If outputs are very similar: Small numerical differences (expected)")
    print("- If outputs differ significantly: Different behavior (investigate)")
    
    # Test with a single forward pass for gradient computation
    print(f"\n🔄 Testing gradient computation...")
    
    for name, model in models.items():
        if name in successful_models:
            try:
                model.train()
                output = model(g, features)
                loss = output.sum()
                loss.backward()
                
                # Check gradient norms
                total_grad_norm = 0.0
                param_count = 0
                
                for param in model.parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.norm().item()
                        total_grad_norm += param_grad_norm ** 2
                        param_count += 1
                
                total_grad_norm = total_grad_norm ** 0.5
                
                print(f"   {name}: Gradient norm = {total_grad_norm:.6f}, Params with grad = {param_count}")
                
                # Clear gradients
                model.zero_grad()
                
            except Exception as e:
                print(f"   {name}: Gradient computation failed - {e}")
    
    return results

def test_weight_copying():
    """Test the weight copying mechanism"""
    print("\n🔧 Testing Weight Copying Mechanism:")
    print("-" * 40)
    
    # Create two models
    set_deterministic_seed(42)
    model1 = SAGE(128, 64, 2, 10, maxk=32).cuda()
    
    set_deterministic_seed(123)  # Different seed
    model2 = HybridMaxKSAGE(128, 64, 2, 10, maxk=32).cuda()
    
    # Check they're different initially
    param1 = dict(model1.named_parameters())
    param2 = dict(model2.named_parameters())
    
    if 'lin_in.weight' in param1 and 'lin_in.weight' in param2:
        initial_diff = torch.abs(param1['lin_in.weight'] - param2['lin_in.weight']).max().item()
        print(f"Initial difference in lin_in.weight: {initial_diff:.6f}")
    
    # Copy weights
    copied = copy_model_weights(model1, model2)
    
    # Check they're similar now
    param2_after = dict(model2.named_parameters())
    if 'lin_in.weight' in param1 and 'lin_in.weight' in param2_after:
        final_diff = torch.abs(param1['lin_in.weight'] - param2_after['lin_in.weight']).max().item()
        print(f"Final difference in lin_in.weight: {final_diff:.6f}")
    
    print(f"Weight copying {'successful' if final_diff < 1e-6 else 'failed'}")

if __name__ == "__main__":
    # Run the comparison
    results = compare_models_side_by_side()
    
    # Test weight copying mechanism
    test_weight_copying()
    
    print(f"\n🎯 Conclusion:")
    print("If models show nearly identical outputs with same initialization,")
    print("then the normalization fix is working correctly and the models")
    print("are mathematically equivalent.")
