#!/usr/bin/env python3
"""
Fixed setup script to build direct CUDA kernel bindings for MaxK-GNN
Addresses compilation issues with includes and visibility warnings
"""

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os
import glob

def get_cuda_arch():
    """Get CUDA architecture for compilation"""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return [f'{major}.{minor}']
    return ['7.5', '8.0', '8.6']  # Common architectures

def check_required_files():
    """Check if all required kernel files exist"""
    required_files = [
        'kernels/spmm_maxk.cu',
        'kernels/spmm_maxk_backward.cu', 
        'kernels/spmm_maxk.h',
        'kernels/spmm_maxk_backward.h',
        'kernels/spmm_base.h',
        'kernels/data.h',
        'kernels/util.h',
    ]
    
    # Check for either the original or fixed bindings file
    bindings_files = ['cuda_kernel_bindings_fixed.cpp', 'cuda_kernel_bindings.cpp']
    bindings_found = False
    bindings_file = None
    for bf in bindings_files:
        if os.path.exists(bf):
            bindings_found = True
            bindings_file = bf
            break
    
    if not bindings_found:
        required_files.append('cuda_kernel_bindings_fixed.cpp')
    
    required_files.append('cuda_kernel_wrappers.cu')
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        print("\n💡 Ensure all kernel files are present:")
        print("   - Copy spmm_maxk.cu and spmm_maxk_backward.cu from kernels/")
        print("   - Create cuda_kernel_bindings_fixed.cpp from artifact")
        print("   - Create cuda_kernel_wrappers.cu from artifact") 
        return False, None
    
    return True, bindings_file

def get_compile_args():
    """Get compilation arguments with fixes for common issues"""
    cuda_arches = get_cuda_arch()
    
    # NVCC arguments - more conservative to avoid issues
    nvcc_args = [
        '-O3',
        '--expt-relaxed-constexpr',
        '--use_fast_math',
        '-std=c++17',  # Use C++14 instead of C++17 for better compatibility
        '--compiler-options=-fPIC,-fvisibility=hidden',  # Hide symbols to avoid visibility warnings
        '-Xcompiler=-fno-strict-aliasing'
    ]
    
    # Add architecture flags
    for arch in cuda_arches:
        clean_arch = arch.replace('.', '')
        nvcc_args.extend([f'-gencode=arch=compute_{clean_arch},code=sm_{clean_arch}'])
    
    # C++ arguments - more conservative
    cxx_args = [
        '-O3',
        '-std=c++17',  # Use C++14 for compatibility
        '-fPIC',
        '-fvisibility=hidden',  # Hide symbols
        '-Wno-attributes',  # Suppress visibility warnings
        '-DTORCH_EXTENSION_NAME=maxk_cuda_kernels'
    ]
    
    return nvcc_args, cxx_args

def create_extension(bindings_file):
    """Create the CUDA extension"""
    
    # Source files
    sources = [
        bindings_file,                      # Python bindings (fixed version)
        'cuda_kernel_wrappers.cu',          # CUDA wrapper functions
        'kernels/spmm_maxk.cu',             # Original MaxK forward kernel
        'kernels/spmm_maxk_backward.cu',     # Original MaxK backward kernel
        'kernels/spmm_cusparse.cu',
        'kernels/maxk_kernel.cu',
    ]
    
    # Include directories
    include_dirs = [
        'kernels/',
        '/usr/local/cuda/include',
        '/opt/cuda/include'  # Alternative CUDA path
    ]
    
    # Libraries
    libraries = ['cusparse', 'cublas']
    
    # Library directories - check multiple common locations
    library_dirs = []
    cuda_lib_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-12.1/lib64',
        '/usr/local/cuda-12.8/lib64',
        '/opt/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ]
    
    for path in cuda_lib_paths:
        if os.path.exists(path):
            library_dirs.append(path)
    
    # Get compilation arguments
    nvcc_args, cxx_args = get_compile_args()
    
    print(f"🔧 Building with CUDA architectures: {get_cuda_arch()}")
    print(f"🔧 Using bindings file: {bindings_file}")
    print(f"🔧 Library directories: {library_dirs}")
    
    # Create extension with error handling
    try:
        ext = CUDAExtension(
            name='maxk_cuda_kernels',
            sources=sources,
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args={
                'cxx': cxx_args,
                'nvcc': nvcc_args
            },
            extra_link_args=['-lcusparse', '-lcublas'],
            verbose=True  # Enable verbose output for debugging
        )
        return ext
    except Exception as e:
        print(f"❌ Failed to create extension: {e}")
        raise

def main():
    """Main setup function"""
    print("🚀 Building Direct MaxK-GNN CUDA Kernel Bindings (Fixed Version)")
    print("=" * 65)
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Cannot build CUDA extensions.")
        return
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU devices: {torch.cuda.device_count()}")
    
    # Check required files
    files_ok, bindings_file = check_required_files()
    if not files_ok:
        print("❌ Cannot proceed without required files")
        return
    
    print("✅ All required files found")
    
    # Create extension
    try:
        extension = create_extension(bindings_file)
        print("✅ Extension configuration created")
    except Exception as e:
        print(f"❌ Failed to create extension: {e}")
        return
    
    # Setup with additional error handling
    try:
        setup(
            name='maxk_cuda_kernels',
            version='1.0.1',
            author='MaxK-GNN Team',
            description='Direct CUDA kernel bindings for MaxK-GNN SPMM operations (Fixed)',
            ext_modules=[extension],
            cmdclass={
                'build_ext': BuildExtension.with_options(
                    use_ninja=False,  # Disable ninja for better error messages
                    max_jobs=1        # Use single job to avoid memory issues
                )
            },
            python_requires='>=3.7',
            install_requires=[
                'torch>=1.12.0',
                'numpy>=1.20.0'
            ],
            zip_safe=False  # Important for CUDA extensions
        )
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\n💡 Try manual build:")
        print("   python setup_direct_kernels_fixed.py clean --all")
        print("   python setup_direct_kernels_fixed.py build_ext --inplace --force")

if __name__ == "__main__":
    main()