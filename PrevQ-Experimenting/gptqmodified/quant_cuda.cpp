#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>


// Declaration for the 8-bit CUDA function (defined in .cu file)
void vecquant8matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros // expects uint8 zeros
);


// C++ Wrapper for the 8-bit CUDA function
void vecquant8matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  // Ensure execution on the correct CUDA device
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));

  // Input validation checks
  TORCH_CHECK(vec.is_cuda(), "Input vector 'vec' must be a CUDA tensor");
  TORCH_CHECK(mat.is_cuda(), "Input matrix 'mat' must be a CUDA tensor");
  TORCH_CHECK(mul.is_cuda(), "Output vector 'mul' must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "Input scales 'scales' must be a CUDA tensor");
  TORCH_CHECK(zeros.is_cuda(), "Input zeros 'zeros' must be a CUDA tensor");

  TORCH_CHECK(vec.dim() >= 1, "Input vector 'vec' must have at least 1 dimension");
  TORCH_CHECK(mat.dim() == 2, "Input matrix 'mat' must have 2 dimensions");
  TORCH_CHECK(mul.dim() == 1, "Output vector 'mul' must have 1 dimension");
  TORCH_CHECK(scales.dim() == 1 || scales.size(0) == mul.size(0), "Scales must be 1D and match output size");
  TORCH_CHECK(zeros.dim() == 1 || zeros.size(0) == mul.size(0), "Zeros must be 1D and match output size");

  TORCH_CHECK(mat.dtype() == torch::kInt32 || mat.dtype() == torch::kUInt32, "Matrix 'mat' must be int32 or uint32");
  TORCH_CHECK(mat.size(1) == mul.size(0), "Matrix columns must match output size");
  TORCH_CHECK(mat.size(1) == scales.size(0), "Matrix columns must match scales size");
  TORCH_CHECK(mat.size(1) == zeros.size(0), "Matrix columns must match zeros size");
  TORCH_CHECK(mat.size(0) * 4 == vec.size(-1), "Packed matrix rows * 4 must match input vector size");

  // Ensure zeros tensor is uint8 as expected by the kernel
  auto zeros_uint8 = zeros.to(torch::kUInt8);

  // Call the CUDA kernel launcher
  vecquant8matmul_cuda(vec, mat, mul, scales, zeros_uint8);
}


// Pybind11 Module Definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant8matmul", &vecquant8matmul, "Vector 8-bit Quantized Matrix Multiplication (CUDA)");
}