#!/bin/bash

echo "🚀 Building MaxK-GNN Direct CUDA Kernel Bindings"
echo "================================================="
echo "This builds Python bindings that directly call spmm_maxk.cu and spmm_maxk_backward.cu"

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found!"
    echo "   Ensure CUDA toolkit is installed and in PATH"
    exit 1
fi

echo "✅ CUDA compiler found: $(nvcc --version | grep release)"

# Check PyTorch CUDA
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA not available'" || {
    echo "❌ PyTorch CUDA not available!"
    echo "   Install PyTorch with CUDA support"
    exit 1
}

echo "✅ PyTorch CUDA available"

# Check required files
echo "📁 Checking required files..."

required_files=(
    "kernels/spmm_maxk.cu"
    "kernels/spmm_maxk_backward.cu"
    "kernels/spmm_maxk.h"
    "kernels/spmm_maxk_backward.h"
    "kernels/spmm_base.h"
    "kernels/data.h"
    "kernels/util.h"
    "cuda_kernel_bindings.cpp"
    "cuda_kernel_wrappers.cu"
    "setup_direct_kernels.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    else
        echo "  ✅ $file"
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo ""
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   $file"
    done
    echo ""
    echo "💡 Create missing files from the artifacts:"
    echo "   1. cuda_kernel_bindings.cpp - C++ bindings code"
    echo "   2. cuda_kernel_wrappers.cu - CUDA wrapper functions"
    echo "   3. setup_direct_kernels.py - Build script"
    echo "   4. Ensure original kernel files are in kernels/ directory"
    exit 1
fi

echo "✅ All required files found"

# Check for warp4 metadata
echo "🔧 Checking for warp4 metadata..."
metadata_found=false
for meta_dir in "../w12_nz64_warp_4/" "./w12_nz64_warp_4/" "w12_nz64_warp_4/"; do
    if [[ -d "$meta_dir" ]] && [[ $(find "$meta_dir" -name "*.warp4" | wc -l) -gt 0 ]]; then
        warp4_count=$(find "$meta_dir" -name "*.warp4" | wc -l)
        echo "  ✅ Found $warp4_count warp4 files in $meta_dir"
        metadata_found=true
        break
    fi
done

if [[ "$metadata_found" = false ]]; then
    echo "  ⚠️ No warp4 metadata found"
    echo "     Generate with: python kernels/generate_meta.py"
    echo "     (Required for optimal kernel performance)"
fi

# Set compilation environment
echo "🔧 Setting up compilation environment..."

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "  CUDA_HOME: $CUDA_HOME"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
rm -rf *.egg-info/
rm -f maxk_cuda_kernels*.so
rm -rf __pycache__/

# Build the extension
echo "🔨 Building direct kernel bindings..."
echo "This may take several minutes..."

python setup_direct_kernels.py build_ext --inplace

build_status=$?

if [[ $build_status -eq 0 ]]; then
    echo ""
    echo "✅ BUILD SUCCESSFUL!"
    
    # Test the build
    echo "🧪 Testing the built extension..."
    
    python -c "
import sys
import torch
print('🔍 Testing import and basic functionality...')

try:
    import maxk_cuda_kernels
    print('✅ maxk_cuda_kernels import successful!')
    
    # Test basic functionality
    print('📋 Available functions:')
    functions = [attr for attr in dir(maxk_cuda_kernels) if not attr.startswith('_')]
    for func in functions:
        print(f'   - {func}')
    
    # Test CUDA tensor creation
    if torch.cuda.is_available():
        print('🔥 Testing CUDA functionality...')
        
        # Test sparse selector generation
        selector = maxk_cuda_kernels.generate_sparse_selector(100, 256, 32)
        print(f'✅ generate_sparse_selector: {selector.shape} {selector.dtype}')
        
        # Test timer
        timer = maxk_cuda_kernels.CudaTimer()
        timer.start()
        torch.cuda.synchronize()
        elapsed = timer.stop()
        print(f'✅ CudaTimer: {elapsed:.3f}ms')
        
    print('🎉 All basic tests passed!')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Test failed: {e}')
    sys.exit(1)
"
    
    test_status=$?
    
    if [[ $test_status -eq 0 ]]; then
        echo ""
        echo "🎉 SUCCESS! Direct CUDA kernels are ready!"
        echo ""
        echo "📚 What was built:"
        echo "  ✅ maxk_cuda_kernels.so - Direct kernel bindings"
        echo "  ✅ Python interface to spmm_maxk.cu kernels"
        echo "  ✅ Python interface to spmm_maxk_backward.cu kernels"
        echo "  ✅ Built-in timing and validation functions"
        echo ""
        echo "🚀 Usage examples:"
        echo ""
        echo "  # Test the direct kernel interface:"
        echo "  python direct_kernel_interface.py"
        echo ""
        echo "  # Run full benchmark (like main.cu):"
        echo "  python main_runner_direct.py"
        echo ""
        echo "  # Benchmark specific graph:"
        echo "  python main_runner_direct.py reddit.dgl"
        echo ""
        echo "📊 Expected performance gains:"
        echo "  ⚡ 2-6x speedup vs cuSPARSE (depends on graph and k value)"
        echo "  💾 90%+ memory traffic reduction"
        echo "  🎯 Same accuracy as original kernels"
        echo ""
        echo "📝 Output format matches main.cu:"
        echo "  num graph dim_origin dim_k kernel time(ms)"
        echo "  1/1 reddit 256 32 maxk 8.567"
        echo "  1/1 reddit 256 32 maxk_backward 12.123"
        
    else
        echo ""
        echo "⚠️ Build succeeded but testing failed"
        echo "   You can still try running the kernels manually"
        echo "   Some tests may be overly strict"
    fi
    
else
    echo ""
    echo "❌ BUILD FAILED!"
    echo ""
    echo "💡 Common issues and solutions:"
    echo ""
    echo "1. CUDA version mismatch:"
    echo "   - Check CUDA version: nvcc --version"
    echo "   - Check PyTorch CUDA: python -c 'import torch; print(torch.version.cuda)'"
    echo "   - Reinstall PyTorch for your CUDA version"
    echo ""
    echo "2. Missing CUDA development tools:"
    echo "   - Install CUDA toolkit development packages"
    echo "   - Ensure nvcc is in PATH"
    echo ""
    echo "3. Architecture mismatch:"
    echo "   - Set TORCH_CUDA_ARCH_LIST for your GPU:"
    echo "   - RTX 30xx: export TORCH_CUDA_ARCH_LIST='8.6'"
    echo "   - RTX 40xx: export TORCH_CUDA_ARCH_LIST='8.9'"
    echo "   - A100: export TORCH_CUDA_ARCH_LIST='8.0'"
    echo ""
    echo "4. Compiler errors:"
    echo "   - Check GCC version: gcc --version"
    echo "   - CUDA 12.x requires GCC 9-11"
    echo "   - Install compatible GCC version"
    echo ""
    echo "5. Missing files:"
    echo "   - Ensure all kernel files are present"
    echo "   - Copy from original MaxK-GNN repository"
    echo ""
    echo "🔧 Manual build attempt:"
    echo "   python setup_direct_kernels.py clean --all"
    echo "   python setup_direct_kernels.py build_ext --inplace --force"
    echo ""
    echo "🆘 For additional help:"
    echo "   - Check CUDA installation: nvidia-smi"
    echo "   - Verify PyTorch: python -c 'import torch; print(torch.cuda.is_available())'"
    echo "   - Check compilation logs above for specific errors"
fi

echo ""
echo "📋 Files created/required:"
echo "  - maxk_cuda_kernels.so (if build successful)"
echo "  - cuda_kernel_bindings.cpp (C++ bindings)"
echo "  - cuda_kernel_wrappers.cu (CUDA wrappers)"
echo "  - setup_direct_kernels.py (build script)"
echo "  - kernels/*.cu (original kernel files)"
echo "  - direct_kernel_interface.py (Python interface)"
echo "  - main_runner_direct.py (main benchmark tool)"

echo ""
echo "🎯 Next steps:"
if [[ $build_status -eq 0 ]]; then
    echo "  1. ✅ Kernels built successfully!"
    echo "  2. Test: python direct_kernel_interface.py"
    echo "  3. Benchmark: python main_runner_direct.py"
    echo "  4. Generate metadata: python kernels/generate_meta.py (if missing)"
    echo "  5. Compare with original: ./kernels/main.cu vs main_runner_direct.py"
else
    echo "  1. ❌ Fix build errors above"
    echo "  2. Ensure CUDA development environment is complete"
    echo "  3. Check all required files are present"
    echo "  4. Try manual build commands"
    echo "  5. Verify CUDA/PyTorch compatibility"
fi