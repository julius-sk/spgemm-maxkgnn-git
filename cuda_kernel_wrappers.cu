// cuda_kernel_wrappers.cu
// Wrapper functions to call the actual CUDA kernels from spmm_maxk.cu and spmm_maxk_backward.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <stdint.h>

// Fix for uint8_t definition
#ifndef uint8_t
typedef unsigned char uint8_t;
#endif

// Include the original kernel declarations
// We need to extract the actual kernel functions from the .cu files

// From spmm_maxk.cu - the actual CUDA kernel
extern __global__ void spmm_kernel_opt2_sparse_v3(
    const int *_warp4, const int *idx, const float *val, 
    const float *vin_data, const uint8_t *vin_selector, float *vout, 
    const int num_v, const int num_e, const int feat_in, 
    const int dim_sparse, const int num_warps
);

// From spmm_maxk_backward.cu - the actual CUDA kernel  
extern __global__ void spmm_kernel_opt2_sparse_backward_v3(
    const int *_warp4, const int *idx, const float *val,
    const float *vin_data, const uint8_t *vin_selector, float *vout,
    const int num_v, const int num_e, const int feat_in,
    const int dim_sparse, const int num_warps
);

// From maxk_kernel.cu - the TopK kernel
extern __global__ void topk(uint8_t *data, uint8_t *value, uint8_t *index, uint k);

// C wrapper functions that can be called from C++
extern "C" {

void spmm_kernel_opt2_sparse_v3_wrapper(
    const int* warp4, const int* idx, const float* val,
    const float* vin_data, const uint8_t* vin_selector, float* vout,
    const int num_v, const int num_e, const int feat_in, 
    const int dim_sparse, const int num_warps,
    dim3 grid, dim3 block, int shared_size
) {
    // Call the actual CUDA kernel from spmm_maxk.cu
    spmm_kernel_opt2_sparse_v3<<<grid, block, shared_size>>>(
        warp4, idx, val, vin_data, vin_selector, vout,
        num_v, num_e, feat_in, dim_sparse, num_warps
    );
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(error));
    }
}

void spmm_kernel_opt2_sparse_backward_v3_wrapper(
    const int* warp4, const int* idx, const float* val,
    const float* vin_data, const uint8_t* vin_selector, float* vout,
    const int num_v, const int num_e, const int feat_in,
    const int dim_sparse, const int num_warps,
    dim3 grid, dim3 block, int shared_size
) {
    // Call the actual CUDA kernel from spmm_maxk_backward.cu
    spmm_kernel_opt2_sparse_backward_v3<<<grid, block, shared_size>>>(
        warp4, idx, val, vin_data, vin_selector, vout,
        num_v, num_e, feat_in, dim_sparse, num_warps
    );
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(error));
    }
}

// Simple TopK kernel wrapper - just add this one function
void topk_kernel_wrapper(
    uint8_t* data, uint8_t* value, uint8_t* index, uint k,
    int N, int dim_origin, dim3 grid, dim3 block, int shared_size
) {
    // Call the actual CUDA kernel from maxk_kernel.cu
    topk<<<grid, block, shared_size>>>(data, value, index, k);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("TopK kernel launch failed: %s\n", cudaGetErrorString(error));
    }
}

} // extern "C"