#include <THC/THC.h>
#include "cuda/sync_bn_kernel.h"

extern THCState *state;

void get_sizes(const THCudaTensor *t, int *N, int *C, int *S)
{
  // Get sizes
  *S = 1;
  *N = THCudaTensor_size(state, t, 0);
  *C = THCudaTensor_size(state, t, 1);
  if (THCudaTensor_nDimension(state, t) > 2)
  { 
    int i = 2;
    for (i = 2; i < THCudaTensor_nDimension(state, t); ++i)
    {
      *S *= THCudaTensor_size(state, t, i);
    }
  }
}

int bn_forward_mean_before_allreduce(const THCudaTensor *bottom, THCudaTensor *mean, const int allreduce_num)
{
  cudaStream_t stream = THCState_getCurrentStream(state);
  int HW, N, C;
  get_sizes(bottom, &N, &C, &HW);

  // Get pointers
  const float *_bottom = THCudaTensor_data(state, bottom);
  float *_mean = THCudaTensor_data(state, mean);

  return bn_forward_mean_before_allreduce_cuda(N, HW, C, allreduce_num, _bottom, _mean, stream);
}

int bn_forward_var_before_allreduce(const THCudaTensor *bottom, const THCudaTensor *mean, THCudaTensor *var, THCudaTensor *top, const int allreduce_num)
{
  cudaStream_t stream = THCState_getCurrentStream(state);
  int HW, N, C;
  get_sizes(bottom, &N, &C, &HW);

  // Get pointers
  const float *_bottom = THCudaTensor_data(state, bottom);
  const float *_mean = THCudaTensor_data(state, mean);
  float *_var = THCudaTensor_data(state, var);
  float *_top = THCudaTensor_data(state, top);

  return bn_forward_var_before_allreduce_cuda(N, HW, C, allreduce_num, _bottom, _mean, _var, _top, stream);
}

int bn_forward_after_allreduce(const THCudaTensor *mean, THCudaTensor *history_mean, const THCudaTensor *var, THCudaTensor *history_var,
                               THCudaTensor *x_norm, THCudaTensor *x_std, const THCudaTensor *scale, const THCudaTensor *shift, THCudaTensor *top,
                               const float var_eps, const float decay)
{
  cudaStream_t stream = THCState_getCurrentStream(state);
  int HW, N, C;
  get_sizes(top, &N, &C, &HW);

  // Get pointers
  const float *_mean = THCudaTensor_data(state, mean);
  float *_history_mean = THCudaTensor_data(state, history_mean);
  const float *_var = THCudaTensor_data(state, var);
  float *_history_var = THCudaTensor_data(state, history_var);
  float *_x_norm = THCudaTensor_data(state, x_norm);
  float *_x_std = THCudaTensor_data(state, x_std);
  const float *_scale = THCudaTensor_data(state, scale);
  const float *_shift = THCudaTensor_data(state, shift);
  float *_top = THCudaTensor_data(state, top);

  return bn_forward_after_allreduce_cuda(N, HW, C, var_eps, decay, _top, _mean, _history_mean, _var, _history_var, _x_norm, _x_std, _scale, _shift, stream);
}

//backward

int bn_backward_before_allreduce(const THCudaTensor *top_diff, const THCudaTensor *x_norm, const THCudaTensor *mean,
                                 const THCudaTensor *x_std, THCudaTensor *bottom_diff, THCudaTensor *local_scale_diff, THCudaTensor *local_shift_diff,
                                 THCudaTensor *scale_diff, THCudaTensor *shift_diff)
{
  cudaStream_t stream = THCState_getCurrentStream(state);
  int HW, N, C;
  get_sizes(top_diff, &N, &C, &HW);

  // Get pointers
  const float *_top_diff = THCudaTensor_data(state, top_diff);
  const float *_x_norm = THCudaTensor_data(state, x_norm);
  const float *_mean = THCudaTensor_data(state, mean);
  const float *_x_std = THCudaTensor_data(state, x_std);
  float *_bottom_diff = THCudaTensor_data(state, bottom_diff);
  float *_local_scale_diff = THCudaTensor_data(state, local_scale_diff);
  float *_local_shift_diff = THCudaTensor_data(state, local_shift_diff);
  float *_scale_diff = THCudaTensor_data(state, scale_diff);
  float *_shift_diff = THCudaTensor_data(state, shift_diff);

  return bn_backward_before_allreduce_cuda(N, HW, C, _top_diff, _x_norm, _mean, _x_std, _bottom_diff, _local_scale_diff, _local_shift_diff, _scale_diff, _shift_diff, stream);
}

int bn_backward_after_allreduce(const THCudaTensor *top_diff, const THCudaTensor *x_norm,
                                const THCudaTensor *local_scale_diff, const THCudaTensor *local_shift_diff,
                                const THCudaTensor *scale_data, const THCudaTensor *x_std, THCudaTensor *bottom_diff,
                                const int allreduce_num)
{
  cudaStream_t stream = THCState_getCurrentStream(state);
  int HW, N, C;
  get_sizes(top_diff, &N, &C, &HW);

  // Get pointers
  const float *_top_diff = THCudaTensor_data(state, top_diff);
  const float *_x_norm = THCudaTensor_data(state, x_norm);
  const float *_local_scale_diff = THCudaTensor_data(state, local_scale_diff);
  const float *_local_shift_diff = THCudaTensor_data(state, local_shift_diff);
  const float *_scale_data = THCudaTensor_data(state, scale_data);
  const float *_x_std = THCudaTensor_data(state, x_std);
  float *_bottom_diff = THCudaTensor_data(state, bottom_diff);

  return bn_backward_after_allreduce_cuda(N, HW, C, _top_diff, _x_norm, _local_scale_diff, _local_shift_diff, _scale_data, _x_std, _bottom_diff, allreduce_num, stream);
}
