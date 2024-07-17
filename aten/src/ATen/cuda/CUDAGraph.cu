#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/MemPool.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace at::cuda {

namespace {

#if !(defined(USE_ROCM)) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
__global__ void set_conditional_handle_kernel(
    cudaGraphConditionalHandle handle,
    const bool* value) {
  cudaGraphSetConditional(handle, *value);
}
#endif
}

void CUDAGraph::set_conditional_handle(
    cudaGraphConditionalHandle handle,
    const Tensor& scalar_cuda_pred_tensor) {
#if !(defined(USE_ROCM)) && (defined(CUDA_VERSION) && CUDA_VERSION >= 12040)
  set_conditional_handle_kernel<<<1, 1, 0, getCurrentCUDAStream()>>>(
      handle, scalar_cuda_pred_tensor.const_data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  AT_ERROR("not allowed");
  return;
#endif
}

} // namespace at::cuda
