#ifdef __cplusplus
extern "C" {
#endif

#include "sync_bn_kernel.h"

#define THREAD_BLOCK_SIZE 256
#define CUDA_POST_KERNEL_CHECK            \
  {                                       \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess)               \
      return 0;                           \
    else                                  \
      return 1;                           \
  }

__global__ void bn_forward_mean_before_allreduce_kernel(const int num, const int map_size, const int channels,
                                                        float stat_ratio, const float *in, float *mean)
{
  __shared__ float buffer[THREAD_BLOCK_SIZE];
  buffer[threadIdx.x] = 0;
  for (int i = threadIdx.x; i < num * map_size; i += blockDim.x)
  {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    buffer[threadIdx.x] += in[location];
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1)
  {
    if (threadIdx.x < i)
      buffer[threadIdx.x] += buffer[threadIdx.x + i];
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    buffer[0] = buffer[0] * stat_ratio;
    mean[blockIdx.x] = buffer[0];
  }
}

__global__ void bn_forward_var_before_allreduce_kernel(const int num, const int map_size, const int channels,
                                                       float stat_ratio, const float *in, const float *mean, float *var, float *out)
{

  __shared__ float buffer[THREAD_BLOCK_SIZE];
  buffer[threadIdx.x] = 0;
  for (int i = threadIdx.x; i < num * map_size; i += blockDim.x)
  {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    out[location] = in[location] - mean[blockIdx.x];
    //buffer[threadIdx.x] += pow(out[location], (float)2);
    buffer[threadIdx.x] += out[location] * out[location];
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1)
  {
    if (threadIdx.x < i)
      buffer[threadIdx.x] += buffer[threadIdx.x + i];
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    buffer[0] = buffer[0] * stat_ratio;
    var[blockIdx.x] = buffer[0];
  }
}

__global__ void bn_forward_after_allreduce_kernel(const int num, const int map_size, const int channels,
                                                  const float stat_eps, const float decay, float *out,
                                                  const float *mean, float *history_mean, const float *var, float *history_var,
                                                  float *x_norm, float *x_std, const float *scale, const float *shift)
{
  //float temp = pow(var[blockIdx.x] + stat_eps, (float)0.5);
  float temp = sqrt(var[blockIdx.x] + stat_eps);
  float scale_value = scale[blockIdx.x], shift_value = shift[blockIdx.x];
  for (int i = threadIdx.x; i < num * map_size; i += blockDim.x)
  {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    x_norm[location] = out[location] / temp;
    out[location] = out[location] / temp * scale_value + shift_value;
  }
  if (threadIdx.x == 0)
  {
    history_mean[blockIdx.x] += decay * (mean[blockIdx.x] - history_mean[blockIdx.x]);
    history_var[blockIdx.x] += decay * (var[blockIdx.x] - history_var[blockIdx.x]);
    x_std[blockIdx.x] = temp;
  }
}

int bn_forward_mean_before_allreduce_cuda(const int N, const int HW, const int C, const int allreduce_num, const float *bottom,
                                          float *mean, cudaStream_t stream)
{
  bn_forward_mean_before_allreduce_kernel<<<C, THREAD_BLOCK_SIZE, 0, stream>>>(N, HW, C, float(1. / (HW * N * allreduce_num)), bottom, mean);
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;
}

int bn_forward_var_before_allreduce_cuda(const int N, const int HW, const int C, const int allreduce_num, const float *bottom,
                                         const float *mean, float *var, float *top, cudaStream_t stream)
{
  int var_correction_factor = HW * N * allreduce_num - 1;
  if (var_correction_factor == 0)
    var_correction_factor = 1;
  // top data has become top[i] - mean[channel] after this kernel, so input data disappear in inplace BN
  bn_forward_var_before_allreduce_kernel<<<C, THREAD_BLOCK_SIZE, 0, stream>>>(N, HW, C, float(1. / var_correction_factor), bottom, mean, var, top);
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;
}

int bn_forward_after_allreduce_cuda(const int N, const int HW, const int C, const float var_eps, const float decay,
                                    float *top, const float *mean, float *history_mean, const float *var, float *history_var,
                                    float *x_norm, float *x_std, const float *scale, const float *shift, cudaStream_t stream)
{

  bn_forward_after_allreduce_kernel<<<C, THREAD_BLOCK_SIZE, 0, stream>>>(N, HW, C, var_eps, decay, top,
                                                                         mean, history_mean, var, history_var,
                                                                         x_norm, x_std, scale, shift);
  CUDA_POST_KERNEL_CHECK;
}

// backward

__global__ void bn_backward_before_allreduce_kernel(const int num, const int map_size, const int channels,
                                                    const float *in, const float *x_norm, const float *mean,
                                                    const float *x_std, float *out, float *local_scale_diff, float *local_shift_diff,
                                                    float *scale_diff, float *shift_diff)
{
  __shared__ float buffer_scale_diff[THREAD_BLOCK_SIZE];
  __shared__ float buffer_shift_diff[THREAD_BLOCK_SIZE];
  buffer_scale_diff[threadIdx.x] = 0;
  buffer_shift_diff[threadIdx.x] = 0;
  for (int i = threadIdx.x; i < num * map_size; i += blockDim.x)
  {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    buffer_scale_diff[threadIdx.x] += (in[location] * x_norm[location]);
    buffer_shift_diff[threadIdx.x] += in[location];
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1)
  {
    if (threadIdx.x < i)
    {
      buffer_scale_diff[threadIdx.x] += buffer_scale_diff[threadIdx.x + i];
      buffer_shift_diff[threadIdx.x] += buffer_shift_diff[threadIdx.x + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    local_scale_diff[blockIdx.x] = buffer_scale_diff[0];
    local_shift_diff[blockIdx.x] = buffer_shift_diff[0];
    scale_diff[blockIdx.x] += buffer_scale_diff[0];
    shift_diff[blockIdx.x] += buffer_shift_diff[0];
  }
}

__global__ void bn_backward_after_allreduce_kernel(const int num, const int map_size, const int channels,
                                                   const float *in, const float *x_norm, const float *local_scale_diff, const float *local_shift_diff,
                                                   const float *scale_data, const float *x_std, float *out, const int num_thread)
{
  for (int i = threadIdx.x; i < num * map_size; i += blockDim.x)
  {
    int location = i / map_size * map_size * channels + (i % map_size) + blockIdx.x * map_size;
    out[location] = scale_data[blockIdx.x] * (in[location] - (x_norm[location] * local_scale_diff[blockIdx.x] + local_shift_diff[blockIdx.x]) / (num * map_size * num_thread)) / x_std[blockIdx.x];
  }
}

int bn_backward_before_allreduce_cuda(const int N, const int HW, const int C,
                                      const float *top_diff, const float *x_norm, const float *mean,
                                      const float *x_std, float *bottom_diff, float *local_scale_diff, float *local_shift_diff,
                                      float *scale_diff, float *shift_diff, cudaStream_t stream)
{

  bn_backward_before_allreduce_kernel<<<C, THREAD_BLOCK_SIZE, 0, stream>>>(N, HW, C, top_diff, x_norm, mean,
                                                                           x_std, bottom_diff, local_scale_diff, local_shift_diff,
                                                                           scale_diff, shift_diff);
  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;
}

int bn_backward_after_allreduce_cuda(const int N, const int HW, const int C,
                                     const float *top_diff, const float *x_norm,
                                     const float *local_scale_diff, const float *local_shift_diff,
                                     const float *scale_data, const float *x_std, float *bottom_diff, const int allreduce_num, cudaStream_t stream)
{
  bn_backward_after_allreduce_kernel<<<C, THREAD_BLOCK_SIZE, 0, stream>>>(N, HW, C, top_diff, x_norm, local_scale_diff, local_shift_diff,
                                                                          scale_data, x_std, bottom_diff, allreduce_num);

  cudaDeviceSynchronize();
  CUDA_POST_KERNEL_CHECK;
}

#ifdef __cplusplus
}
#endif