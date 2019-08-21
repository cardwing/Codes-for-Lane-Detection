int bn_forward_mean_before_allreduce(const THCudaTensor *bottom, THCudaTensor *mean, const int allreduce_num);

int bn_forward_var_before_allreduce(const THCudaTensor *bottom, const THCudaTensor *mean, THCudaTensor *var, THCudaTensor *top, const int allreduce_num);

int bn_forward_after_allreduce(const THCudaTensor *mean, THCudaTensor *history_mean, const THCudaTensor *var, THCudaTensor *history_var,
							   THCudaTensor *x_norm, THCudaTensor *x_std, const THCudaTensor *scale, const THCudaTensor *shift, THCudaTensor *top,
							   const float var_eps, const float decay);

int bn_backward_before_allreduce(const THCudaTensor *top_diff, const THCudaTensor *x_norm, const THCudaTensor *mean,
								 const THCudaTensor *x_std, THCudaTensor *bottom_diff, THCudaTensor *local_scale_diff, THCudaTensor *local_shift_diff,
								 THCudaTensor *scale_diff, THCudaTensor *shift_diff);

int bn_backward_after_allreduce(const THCudaTensor *top_diff, const THCudaTensor *x_norm,
								const THCudaTensor *local_scale_diff, const THCudaTensor *local_shift_diff,
								const THCudaTensor *scale_data, const THCudaTensor *x_std, THCudaTensor *bottom_diff,
								const int allreduce_num);