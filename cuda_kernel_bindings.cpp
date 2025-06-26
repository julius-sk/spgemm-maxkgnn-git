#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdint>

// Forward declarations of CUDA kernels from spmm_maxk.cu and spmm_maxk_backward.cu
extern "C" {
    // From spmm_maxk.cu
    void spmm_kernel_opt2_sparse_v3_wrapper(
        const int* warp4, const int* idx, const float* val,
        const float* vin_data, const uint8_t* vin_selector, float* vout,
        const int num_v, const int num_e, const int feat_in, 
        const int dim_sparse, const int num_warps,
        dim3 grid, dim3 block, int shared_size
    );
    
    // From spmm_maxk_backward.cu  
    void spmm_kernel_opt2_sparse_backward_v3_wrapper(
        const int* warp4, const int* idx, const float* val,
        const float* vin_data, const uint8_t* vin_selector, float* vout,
        const int num_v, const int num_e, const int feat_in,
        const int dim_sparse, const int num_warps,
        dim3 grid, dim3 block, int shared_size
    );
    
    // ADD: TopK kernel from maxk_kernel.cu
    void topk_kernel_wrapper(
        uint8_t* data, uint8_t* value, uint8_t* index, uint k,
        int N, int dim_origin, dim3 grid, dim3 block, int shared_size
    );
}

// Declare cuSPARSE function (C++ linkage, not extern "C")
double spmm_cusparse(int *ptr, int *idx, float *val, float *vin, float *vout, 
                    int num_v, int num_e, int dim, int times);

// CUDA kernel wrapper functions
torch::Tensor spmm_maxk_forward(
    torch::Tensor warp4_metadata,     // Warp scheduling metadata
    torch::Tensor indices,            // Graph indices (CSR format)
    torch::Tensor values,             // Edge values
    torch::Tensor input_data,         // Dense input features (sparse representation)
    torch::Tensor sparse_selector,    // Sparse selector indices  
    int num_warps,                    // Number of warps
    int dim_sparse                    // Sparse dimension (k)
) {
    // Validate inputs
    TORCH_CHECK(warp4_metadata.is_cuda(), "warp4_metadata must be CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA tensor");
    TORCH_CHECK(input_data.is_cuda(), "input_data must be CUDA tensor");
    TORCH_CHECK(sparse_selector.is_cuda(), "sparse_selector must be CUDA tensor");
    
    TORCH_CHECK(warp4_metadata.dtype() == torch::kInt32, "warp4_metadata must be int32");
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(values.dtype() == torch::kFloat32, "values must be float32");
    TORCH_CHECK(input_data.dtype() == torch::kFloat32, "input_data must be float32");
    TORCH_CHECK(sparse_selector.dtype() == torch::kUInt8, "sparse_selector must be uint8");
    
    // Get dimensions
    int num_v = input_data.size(0);
    int sparse_dim = input_data.size(1);  // This is dim_sparse (32)
    int num_e = indices.size(0);
    
    // FIXED: Create output tensor with FULL dimensions (256), not sparse dimensions
    const int FULL_DIM = 256;  // Full feature dimension
    auto output = torch::zeros({num_v, FULL_DIM}, 
                              torch::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(input_data.device()));
    
    // Calculate grid and block dimensions (from spmm_maxk.cu logic)
    const int WARPS_PER_BLOCK = 12;
    const int EXT_WARP_DIM = 32;
    
    int block_num = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(block_num);
    dim3 block(WARPS_PER_BLOCK * EXT_WARP_DIM);
    
    // Calculate shared memory size - use FULL dimension for shared memory
    int shared_size = WARPS_PER_BLOCK * FULL_DIM * sizeof(float);
    
    // Call CUDA kernel
    spmm_kernel_opt2_sparse_v3_wrapper(
        warp4_metadata.data_ptr<int>(),
        indices.data_ptr<int>(),
        values.data_ptr<float>(),
        input_data.data_ptr<float>(),
        sparse_selector.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        num_v, num_e, FULL_DIM, dim_sparse, num_warps,  // Pass FULL_DIM as feat_in
        grid, block, shared_size
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(error));
    
    return output;
}

torch::Tensor spmm_maxk_backward(
    torch::Tensor warp4_metadata,     // Warp scheduling metadata
    torch::Tensor indices,            // Graph indices (CSR format)  
    torch::Tensor values,             // Edge values
    torch::Tensor grad_output,        // Gradient from next layer
    torch::Tensor sparse_selector,    // Sparse selector indices
    int num_warps,                    // Number of warps
    int dim_sparse                    // Sparse dimension (k)
) {
    // Validate inputs
    TORCH_CHECK(warp4_metadata.is_cuda(), "warp4_metadata must be CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA tensor");
    TORCH_CHECK(sparse_selector.is_cuda(), "sparse_selector must be CUDA tensor");
    
    // Get dimensions
    int num_v = grad_output.size(0);
    int feat_in = grad_output.size(1);  // Full dimension (256)
    int num_e = indices.size(0);
    
    // Create output tensor (sparse format) - CORRECT
    auto grad_input = torch::zeros({num_v, dim_sparse},
                                  torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(grad_output.device()));
    
    // Calculate grid and block dimensions
    const int WARPS_PER_BLOCK = 12;
    const int EXT_WARP_DIM = 32;
    
    int block_num = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(block_num);
    dim3 block(WARPS_PER_BLOCK * EXT_WARP_DIM);
    
    // Calculate shared memory size
    int shared_size = WARPS_PER_BLOCK * feat_in * sizeof(float);
    
    // Call CUDA kernel
    spmm_kernel_opt2_sparse_backward_v3_wrapper(
        warp4_metadata.data_ptr<int>(),
        indices.data_ptr<int>(),
        values.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        sparse_selector.data_ptr<uint8_t>(),
        grad_input.data_ptr<float>(),
        num_v, num_e, feat_in, dim_sparse, num_warps,
        grid, block, shared_size
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(error));
    
    return grad_input;
}

// ADD: TopK kernel wrapper functions
std::tuple<torch::Tensor, torch::Tensor> cuda_topk_maxk(
    torch::Tensor input, int k) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8 tensor");
    TORCH_CHECK(k > 0 && k <= input.size(1), "Invalid k value");
    
    int N = input.size(0);
    int dim_origin = input.size(1);
    
    // Create output tensors
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    torch::Tensor values = torch::zeros({N, k}, options);
    torch::Tensor indices = torch::zeros({N, k}, options);
    
    // Calculate grid and block dimensions (from maxk_kernel.cu constants)
    const int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;  // 32 threads per warp
    int grid_size = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int shared_size = WARPS_PER_BLOCK * (dim_origin + 2 * k + 1);
    
    dim3 grid(grid_size);
    dim3 block(block_size);
    
    // Launch the CUDA kernel via wrapper
    topk_kernel_wrapper(
        input.data_ptr<uint8_t>(),
        values.data_ptr<uint8_t>(),
        indices.data_ptr<uint8_t>(),
        k, N, dim_origin, grid, block, shared_size
    );
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
    
    return std::make_tuple(values, indices);
}

std::tuple<torch::Tensor, torch::Tensor> cuda_topk_maxk_float(
    torch::Tensor input, int k) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(k > 0 && k <= input.size(1), "Invalid k value");
    
    // If input is float, convert to uint8 for kernel processing
    torch::Tensor input_uint8;
    if (input.dtype() == torch::kFloat32) {
        // Scale and convert to uint8 (0-255 range)
        auto normalized = torch::clamp((input * 255.0).round(), 0, 255);
        input_uint8 = normalized.to(torch::kUInt8);
    } else if (input.dtype() == torch::kUInt8) {
        input_uint8 = input;
    } else {
        TORCH_CHECK(false, "Input must be float32 or uint8");
    }
    cout << "++++++++++++++++++++++++++++" << endl;
    // Run the uint8 kernel
    auto [values_uint8, indices_uint8] = cuda_topk_maxk(input_uint8, k);
    
    // Convert results back to appropriate types
    torch::Tensor values, indices;
    
    if (input.dtype() == torch::kFloat32) {
        // Convert values back to float
        values = values_uint8.to(torch::kFloat32) / 255.0;
        indices = indices_uint8.to(torch::kInt32);
    } else {
        values = values_uint8;
        indices = indices_uint8.to(torch::kInt32);
    }
    
    return std::make_tuple(values, indices);
}

std::tuple<torch::Tensor, torch::Tensor> prepare_cbsr_format_maxk(
    torch::Tensor features, int maxk) {
    
    TORCH_CHECK(features.is_cuda(), "Features must be on CUDA");
    TORCH_CHECK(features.dim() == 2, "Features must be 2D tensor");
    TORCH_CHECK(maxk > 0 && maxk <= features.size(1), "Invalid maxk value");
    
    // Use our enhanced TopK function
    auto [sparse_data, sparse_indices] = cuda_topk_maxk_float(features, maxk);
    
    return std::make_tuple(sparse_data, sparse_indices);
}

torch::Tensor cusparse_spmm_wrapper(
    torch::Tensor indptr,
    torch::Tensor indices, 
    torch::Tensor values,
    torch::Tensor input_features,
    bool timing = false) {
    
    TORCH_CHECK(indptr.is_cuda() && indptr.is_contiguous(), "indptr must be CUDA and contiguous");
    TORCH_CHECK(indices.is_cuda() && indices.is_contiguous(), "indices must be CUDA and contiguous");
    TORCH_CHECK(values.is_cuda() && values.is_contiguous(), "values must be CUDA and contiguous");
    TORCH_CHECK(input_features.is_cuda() && input_features.is_contiguous(), "input_features must be CUDA and contiguous");
    
    int num_v = indptr.size(0) - 1;
    int num_e = indices.size(0);
    int dim = input_features.size(1);
    
    // Create output tensor
    auto output = torch::zeros_like(input_features);
    
    // Call the actual cuSPARSE kernel
    double exec_time = spmm_cusparse(
        indptr.data_ptr<int>(),
        indices.data_ptr<int>(),
        values.data_ptr<float>(),
        input_features.data_ptr<float>(),
        output.data_ptr<float>(),
        num_v, num_e, dim,
        timing ? 10 : 0  // number of timing runs
    );
    
    return output;
}

// FIXED: Utility function to load warp4 metadata with correct path
torch::Tensor load_warp4_metadata(const std::string& graph_name, 
                                  int num_warps = 12, int warp_max_nz = 64) {
    // FIXED: Correct path for metadata
    std::string meta_dir = "kernels/w" + std::to_string(num_warps) + "_nz" + 
                          std::to_string(warp_max_nz) + "_warp_4/";
    std::string warp4_file = meta_dir + graph_name + ".warp4";
    
    // Read binary file
    std::ifstream file(warp4_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open warp4 file: " + warp4_file);
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read data
    size_t num_elements = static_cast<size_t>(file_size) / sizeof(int32_t);
    std::vector<int32_t> warp4_data(num_elements);
    file.read(reinterpret_cast<char*>(warp4_data.data()), static_cast<std::streamsize>(file_size));
    file.close();
    
    // Convert to tensor
    auto tensor = torch::from_blob(warp4_data.data(), {static_cast<long>(warp4_data.size())}, 
                                  torch::TensorOptions().dtype(torch::kInt32))
                  .clone().cuda();
    
    return tensor;
}

// Generate sparse selector (replicates main.cu logic)
torch::Tensor generate_sparse_selector(int num_v, int dim_origin, int dim_sparse) {
    auto selector = torch::zeros({num_v, dim_sparse}, 
                                torch::TensorOptions()
                                .dtype(torch::kUInt8)
                                .device(torch::kCUDA));
    
    // Use same random seed as main.cu
    torch::manual_seed(123);
    
    for (int i = 0; i < num_v; i++) {
        auto indices = torch::randperm(dim_origin, 
                                     torch::TensorOptions()
                                     .dtype(torch::kInt64)
                                     .device(torch::kCUDA))
                      .slice(0, 0, dim_sparse)
                      .to(torch::kUInt8);
        selector[i] = indices;
    }
    
    return selector;
}

// Simple timing class
class SimpleCudaTimer {
public:
    SimpleCudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~SimpleCudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        cudaEventRecord(start_);
    }
    
    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return elapsed; // milliseconds
    }
    
private:
    cudaEvent_t start_, stop_;
};

// Benchmark function that mimics main.cu timing
std::vector<float> benchmark_spmm_maxk(
    torch::Tensor warp4_metadata,
    torch::Tensor indices,
    torch::Tensor values, 
    torch::Tensor input_data,
    torch::Tensor sparse_selector,
    int num_warps,
    int dim_sparse,
    int num_runs = 4
) {
    std::vector<float> times;
    SimpleCudaTimer timer;
    
    // Warmup runs
    for (int i = 0; i < num_runs; i++) {
        spmm_maxk_forward(warp4_metadata, indices, values, input_data, 
                         sparse_selector, num_warps, dim_sparse);
    }
    cudaDeviceSynchronize();
    
    // Timing runs
    for (int i = 0; i < num_runs; i++) {
        timer.start();
        auto result = spmm_maxk_forward(warp4_metadata, indices, values, input_data,
                                       sparse_selector, num_warps, dim_sparse);
        float elapsed = timer.stop();
        times.push_back(elapsed);
    }
    
    return times;
}

// Validation function
bool validate_spmm_maxk(
    torch::Tensor warp4_metadata,
    torch::Tensor indices,
    torch::Tensor values,
    torch::Tensor input_data,
    torch::Tensor sparse_selector,
    torch::Tensor reference_output,
    int num_warps,
    int dim_sparse,
    float tolerance = 0.001f
) {
    auto maxk_output = spmm_maxk_forward(warp4_metadata, indices, values, input_data,
                                        sparse_selector, num_warps, dim_sparse);
    
    auto diff = torch::abs(maxk_output - reference_output);
    float max_diff = torch::max(diff).item<float>();
    float avg_diff = torch::mean(diff).item<float>();
    
    std::cout << "Validation - Max diff: " << max_diff 
              << ", Avg diff: " << avg_diff << std::endl;
    
    return avg_diff < tolerance;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Direct CUDA kernel bindings for MaxK-GNN SPMM operations";
    
    // Main kernel functions
    m.def("spmm_maxk_forward", &spmm_maxk_forward, 
          "MaxK-GNN forward SPMM kernel",
          pybind11::arg("warp4_metadata"), pybind11::arg("indices"), pybind11::arg("values"),
          pybind11::arg("input_data"), pybind11::arg("sparse_selector"), 
          pybind11::arg("num_warps"), pybind11::arg("dim_sparse"));
    
    m.def("spmm_maxk_backward", &spmm_maxk_backward,
          "MaxK-GNN backward SPMM kernel", 
          pybind11::arg("warp4_metadata"), pybind11::arg("indices"), pybind11::arg("values"),
          pybind11::arg("grad_output"), pybind11::arg("sparse_selector"),
          pybind11::arg("num_warps"), pybind11::arg("dim_sparse"));
    
    // ADD: TopK kernel functions
    m.def("cuda_topk_maxk", &cuda_topk_maxk,
          "Fast CUDA TopK kernel for uint8 tensors",
          pybind11::arg("input"), pybind11::arg("k"));
    
    m.def("cuda_topk_maxk_float", &cuda_topk_maxk_float,
          "Fast CUDA TopK kernel for float tensors",
          pybind11::arg("input"), pybind11::arg("k"));
    
    m.def("prepare_cbsr_format_maxk", &prepare_cbsr_format_maxk,
          "Prepare CBSR format using MaxK TopK kernel",
          pybind11::arg("features"), pybind11::arg("maxk"));
    
    // Utility functions
    m.def("load_warp4_metadata", &load_warp4_metadata,
          "Load warp4 metadata from file",
          pybind11::arg("graph_name"), pybind11::arg("num_warps") = 12, pybind11::arg("warp_max_nz") = 64);
    
    m.def("generate_sparse_selector", &generate_sparse_selector,
          "Generate sparse selector tensor",
          pybind11::arg("num_v"), pybind11::arg("dim_origin"), pybind11::arg("dim_sparse"));
    
    // Benchmarking functions
    m.def("benchmark_spmm_maxk", &benchmark_spmm_maxk,
          "Benchmark MaxK-GNN SPMM kernel",
          pybind11::arg("warp4_metadata"), pybind11::arg("indices"), pybind11::arg("values"),
          pybind11::arg("input_data"), pybind11::arg("sparse_selector"),
          pybind11::arg("num_warps"), pybind11::arg("dim_sparse"), pybind11::arg("num_runs") = 4);
    
    m.def("validate_spmm_maxk", &validate_spmm_maxk,
          "Validate MaxK-GNN kernel against reference",
          pybind11::arg("warp4_metadata"), pybind11::arg("indices"), pybind11::arg("values"),
          pybind11::arg("input_data"), pybind11::arg("sparse_selector"), pybind11::arg("reference_output"),
          pybind11::arg("num_warps"), pybind11::arg("dim_sparse"), pybind11::arg("tolerance") = 0.001f);

    // FIXED: Correct pybind11 namespace
    m.def("cusparse_spmm", &cusparse_spmm_wrapper, "cuSPARSE SpMM reference",
          pybind11::arg("indptr"), pybind11::arg("indices"), pybind11::arg("values"), 
          pybind11::arg("input_features"), pybind11::arg("timing") = false);

    // Expose SimpleCudaTimer class
    pybind11::class_<SimpleCudaTimer>(m, "CudaTimer")
        .def(pybind11::init<>())
        .def("start", &SimpleCudaTimer::start)
        .def("stop", &SimpleCudaTimer::stop);
}