#!/bin/bash

# MaxK-GNN Training Scripts with SpGEMM Integration
# Updated versions of the original training scripts to use MaxK CUDA kernels

# Reddit dataset with MaxK kernels
function train_reddit_maxk_kernels() {
    if [ "$#" -ne 4 ]; then
        echo "Usage: train_reddit_maxk_kernels <k> <seed> <gpu> <model>"
        exit 1
    fi
    
    k="$1"
    seed="$2"
    gpu="$3"
    model="$4"
    export dataset=reddit
    
    echo "🚀 Training Reddit with MaxK CUDA kernels (k=${k})"
    
    mkdir -p ./log/${dataset}_seed${seed}/
    nohup python -u maxk_gnn_integrated.py --dataset ${dataset} --model ${model} \
     --hidden_layers 4 --hidden_dim 256 --nonlinear "maxk" --maxk ${k} \
     --dropout 0.5 --norm --w_lr 0.01 --seed ${seed} \
     --use_maxk_kernels --kernel_mode auto --validate_kernels --profile_kernels \
     --path experiment/${dataset}_seed${seed}/${model}_maxk_kernels_${k} --epochs 3000 --gpu ${gpu} \
     > ./log/${dataset}_seed${seed}/${model}_maxk_kernels_${k}.txt &
}

# Flickr dataset with MaxK kernels
function train_flickr_maxk_kernels() {
    if [ "$#" -ne 4 ]; then
        echo "Usage: train_flickr_maxk_kernels <k> <seed> <gpu> <model>"
        exit 1
    fi
    
    k="$1"
    seed="$2"
    gpu="$3"
    model="$4"
    export dataset=flickr
    
    echo "🚀 Training Flickr with MaxK CUDA kernels (k=${k})"
    
    mkdir -p ./log/${dataset}_seed${seed}/
    nohup python -u maxk_gnn_integrated.py --dataset ${dataset} --model ${model} --selfloop \
     --hidden_layers 3 --hidden_dim 256 --nonlinear "maxk" --maxk ${k} \
     --dropout 0.2 --norm --w_lr 0.001 --seed ${seed} \
     --use_maxk_kernels --kernel_mode auto --validate_kernels --profile_kernels \
     --path experiment/${dataset}_seed${seed}/${model}_maxk_kernels_${k} --epochs 400 --gpu ${gpu} \
     > ./log/${dataset}_seed${seed}/${model}_maxk_kernels_${k}.txt &
}

# OGBN-Products dataset with MaxK kernels
function train_ogbn_products_maxk_kernels() {
    if [ "$#" -ne 4 ]; then
        echo "Usage: train_ogbn_products_maxk_kernels <k> <seed> <gpu> <model>"
        exit 1
    fi
    
    k="$1"
    seed="$2"
    gpu="$3"
    model="$4"
    export dataset=ogbn-products
    
    if [ "$model" == "sage" ]; then
        selfloop=""
    else
        selfloop=--selfloop
    fi
    
    echo "🚀 Training OGBN-Products with MaxK CUDA kernels (k=${k})"
    
    mkdir -p ./log/${dataset}_seed${seed}/
    nohup python -u maxk_gnn_integrated.py --dataset ${dataset} --model ${model} ${selfloop} \
     --hidden_layers 3 --hidden_dim 256 --nonlinear "maxk" --maxk ${k} \
     --dropout 0.5 --norm --w_lr 0.003 --seed ${seed} \
     --use_maxk_kernels --kernel_mode auto --validate_kernels --profile_kernels \
     --path experiment/${dataset}_seed${seed}/${model}_maxk_kernels_${k} --epochs 500 --gpu ${gpu} <<< "y" \
     > ./log/${dataset}_seed${seed}/${model}_maxk_kernels_${k}.txt &
}

# Comparison script: Original vs MaxK kernels
function compare_maxk_performance() {
    if [ "$#" -ne 4 ]; then
        echo "Usage: compare_maxk_performance <dataset> <k> <seed> <gpu>"
        exit 1
    fi
    
    dataset="$1"
    k="$2"
    seed="$3"
    gpu="$4"
    model="sage"  # Default to SAGE for comparison
    
    echo "🔬 Performance Comparison: Original vs MaxK Kernels"
    echo "Dataset: ${dataset}, k=${k}, seed=${seed}, GPU=${gpu}"
    
    # Original training (DGL)
    echo "📊 Running original DGL training..."
    mkdir -p ./log/${dataset}_seed${seed}/
    python -u maxk_gnn_dgl.py --dataset ${dataset} --model ${model} \
     --hidden_layers 3 --hidden_dim 256 --nonlinear "maxk" --maxk ${k} \
     --dropout 0.5 --norm --w_lr 0.01 --seed ${seed} \
     --path experiment/${dataset}_seed${seed}/${model}_original_${k} --epochs 100 --gpu ${gpu} \
     > ./log/${dataset}_seed${seed}/${model}_original_${k}_comparison.txt
    
    echo "⚡ Running MaxK kernel training..."
    # MaxK kernel training
    python -u maxk_gnn_integrated.py --dataset ${dataset} --model ${model} \
     --hidden_layers 3 --hidden_dim 256 --nonlinear "maxk" --maxk ${k} \
     --dropout 0.5 --norm --w_lr 0.01 --seed ${seed} \
     --use_maxk_kernels --kernel_mode auto --validate_kernels --profile_kernels \
     --path experiment/${dataset}_seed${seed}/${model}_maxk_kernels_${k} --epochs 100 --gpu ${gpu} \
     > ./log/${dataset}_seed${seed}/${model}_maxk_kernels_${k}_comparison.txt
    
    echo "📈 Comparison complete! Check log files for timing results."
}

# Batch training script for multiple k values
function batch_train_maxk() {
    if [ "$#" -ne 4 ]; then
        echo "Usage: batch_train_maxk <dataset> <model> <seed> <gpu>"
        exit 1
    fi
    
    dataset="$1"
    model="$2"
    seed="$3"
    gpu="$4"
    
    k_values=(16 32 64)
    
    echo "🔄 Batch training with MaxK kernels"
    echo "Dataset: ${dataset}, Model: ${model}, Seed: ${seed}, GPU: ${gpu}"
    echo "K values: ${k_values[@]}"
    
    for k in "${k_values[@]}"; do
        echo "📊 Training with k=${k}..."
        
        case ${dataset} in
            "reddit")
                train_reddit_maxk_kernels ${k} ${seed} ${gpu} ${model}
                ;;
            "flickr")
                train_flickr_maxk_kernels ${k} ${seed} ${gpu} ${model}
                ;;
            "ogbn-products")
                train_ogbn_products_maxk_kernels ${k} ${seed} ${gpu} ${model}
                ;;
            *)
                echo "⚠️ Dataset ${dataset} not supported in batch mode"
                ;;
        esac
        
        # Wait a bit between runs
        sleep 5
    done
    
    echo "✅ Batch training submitted for all k values"
}

# Validation script to check kernel correctness
function validate_maxk_kernels() {
    if [ "$#" -ne 2 ]; then
        echo "Usage: validate_maxk_kernels <dataset> <gpu>"
        exit 1
    fi
    
    dataset="$1"
    gpu="$2"
    
    echo "🧪 Validating MaxK kernel correctness"
    echo "Dataset: ${dataset}, GPU: ${gpu}"
    
    python -c "
import sys
sys.path.append('.')
from maxk_spgemm_function import test_maxk_spgemm_function
from maxk_models_integrated import test_maxk_sage
from direct_kernel_interface import test_direct_kernels

print('🔍 Testing MaxK SpGEMM function...')
test_maxk_spgemm_function()

print('🔍 Testing MaxK SAGE integration...')
test_maxk_sage()

print('🔍 Testing direct kernel interface...')
test_direct_kernels()

print('✅ All validation tests completed!')
"
}

# Performance profiling script
function profile_maxk_kernels() {
    if [ "$#" -ne 3 ]; then
        echo "Usage: profile_maxk_kernels <dataset> <k> <gpu>"
        exit 1
    fi
    
    dataset="$1"
    k="$2"
    gpu="$3"
    
    echo "📊 Profiling MaxK kernel performance"
    echo "Dataset: ${dataset}, k=${k}, GPU: ${gpu}"
    
    # Short training run with detailed profiling
    python -u maxk_gnn_integrated.py --dataset ${dataset} --model sage \
     --hidden_layers 2 --hidden_dim 128 --nonlinear "maxk" --maxk ${k} \
     --dropout 0.5 --norm --w_lr 0.01 --seed 42 \
     --use_maxk_kernels --kernel_mode auto --validate_kernels --profile_kernels \
     --epochs 10 --gpu ${gpu} \
     --path ./profile_maxk_${dataset}_k${k}
}

# Setup script to ensure everything is ready
function setup_maxk_training() {
    echo "🔧 Setting up MaxK training environment..."
    
    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        echo "❌ CUDA not found! Install CUDA toolkit."
        exit 1
    fi
    
    # Check Python packages
    python -c "import torch; assert torch.cuda.is_available()" || {
        echo "❌ PyTorch CUDA not available!"
        exit 1
    }
    
    # Check MaxK kernels
    python -c "import maxk_cuda_kernels; print('✅ MaxK kernels available')" || {
        echo "⚠️ MaxK kernels not built. Building now..."
        python setup_direct_kernels.py build_ext --inplace || {
            echo "❌ Failed to build MaxK kernels!"
            exit 1
        }
    }
    
    # Check metadata
    if [ ! -d "kernels/w12_nz64_warp_4" ]; then
        echo "📊 Generating warp4 metadata..."
        cd kernels
        python generate_meta.py
        cd ..
    fi
    
    # Create log directories
    mkdir -p log
    mkdir -p experiment
    
    echo "✅ MaxK training environment ready!"
}

# Help function
function show_help() {
    echo "MaxK-GNN Training Scripts with SpGEMM Integration"
    echo "================================================="
    echo ""
    echo "Setup:"
    echo "  setup_maxk_training                    - Setup training environment"
    echo ""
    echo "Individual Training:"
    echo "  train_reddit_maxk_kernels <k> <seed> <gpu> <model>"
    echo "  train_flickr_maxk_kernels <k> <seed> <gpu> <model>"
    echo "  train_ogbn_products_maxk_kernels <k> <seed> <gpu> <model>"
    echo ""
    echo "Batch Operations:"
    echo "  batch_train_maxk <dataset> <model> <seed> <gpu>"
    echo "  compare_maxk_performance <dataset> <k> <seed> <gpu>"
    echo ""
    echo "Validation & Profiling:"
    echo "  validate_maxk_kernels <dataset> <gpu>"
    echo "  profile_maxk_kernels <dataset> <k> <gpu>"
    echo ""
    echo "Examples:"
    echo "  # Setup environment"
    echo "  setup_maxk_training"
    echo ""
    echo "  # Train Reddit with k=32"
    echo "  train_reddit_maxk_kernels 32 97 0 sage"
    echo ""
    echo "  # Batch train multiple k values"
    echo "  batch_train_maxk reddit sage 97 0"
    echo ""
    echo "  # Compare original vs MaxK kernels"
    echo "  compare_maxk_performance reddit 32 97 0"
    echo ""
    echo "  # Validate kernel correctness"
    echo "  validate_maxk_kernels reddit 0"
}

# Main script logic
if [ "$#" -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    "setup")
        setup_maxk_training
        ;;
    "train_reddit")
        shift
        train_reddit_maxk_kernels "$@"
        ;;
    "train_flickr")
        shift
        train_flickr_maxk_kernels "$@"
        ;;
    "train_ogbn_products")
        shift
        train_ogbn_products_maxk_kernels "$@"
        ;;
    "batch")
        shift
        batch_train_maxk "$@"
        ;;
    "compare")
        shift
        compare_maxk_performance "$@"
        ;;
    "validate")
        shift
        validate_maxk_kernels "$@"
        ;;
    "profile")
        shift
        profile_maxk_kernels "$@"
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use 'help' to see available commands"
        exit 1
        ;;
esac
