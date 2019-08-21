#ifndef __SYNC_BN_KERNEL__
#define __SYNC_BN_KERNEL__

#ifdef __cplusplus
extern "C" {
#endif

int bn_forward_mean_before_allreduce_cuda(const int N, const int HW, const int C, const int allreduce_num, const float *bottom,
										  float *mean, cudaStream_t stream);

int bn_forward_var_before_allreduce_cuda(const int N, const int HW, const int C, const int allreduce_num, const float *bottom,
										 const float *mean, float *var, float *top, cudaStream_t stream);

int bn_forward_after_allreduce_cuda(const int N, const int HW, const int C, const float var_eps, const float decay,
									float *top, const float *mean, float *history_mean, const float *var, float *history_var,
									float *x_norm, float *x_std, const float *scale, const float *shift, cudaStream_t stream);

int bn_backward_before_allreduce_cuda(const int N, const int HW, const int C,
									  const float *top_diff, const float *x_norm, const float *mean,
									  const float *x_std, float *bottom_diff, float *local_scale_diff, float *local_shift_diff,
									  float *scale_diff, float *shift_diff, cudaStream_t stream);

int bn_backward_after_allreduce_cuda(const int N, const int HW, const int C,
									 const float *top_diff, const float *x_norm,
									 const float *local_scale_diff, const float *local_shift_diff,
									 const float *scale_data, const float *x_std, float *bottom_diff, const int allreduce_num, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif //__SYNC_BN_KERNEL__