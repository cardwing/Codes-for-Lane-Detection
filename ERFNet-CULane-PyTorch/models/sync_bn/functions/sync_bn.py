# functions/add.py
import torch
import torch.cuda.nccl as nccl
from torch.autograd import Function
import threading
from .._ext import sync_bn_lib

__all__ = ['sync_batch_norm', 'Synchronize']


class Synchronize:
    has_Listener = False
    device_num = 1
    data_ready = []
    data_list = []
    result_list = []
    result_ready = []

    def init(device_num):
        if Synchronize.has_Listener:
            return
        else:
            Synchronize.has_Listener = True

        Synchronize.device_num = device_num
        Synchronize.data_list = [None] * device_num
        Synchronize.result_list = [None] * device_num
        Synchronize.data_ready = [threading.Event() for _ in range(device_num)]
        Synchronize.result_ready = [threading.Event() for _ in range(device_num)]

        for i in range(Synchronize.device_num):
            Synchronize.data_ready[i].clear()
            Synchronize.result_ready[i].set()

        def _worker():
            while (True):
                for i in range(Synchronize.device_num):
                    Synchronize.data_ready[i].wait()

                total_sum = Synchronize.data_list[0].cpu().clone()
                for i in range(1, Synchronize.device_num):
                    total_sum = total_sum + Synchronize.data_list[i].cpu()

                for i in range(0, Synchronize.device_num):
                    with torch.cuda.device_of(Synchronize.data_list[i]):
                        Synchronize.result_list[i] = total_sum.clone().cuda()

                # nccl.all_reduce_sync(Synchronize.data_list, Synchronize.result_list)

                for i in range(Synchronize.device_num):
                    Synchronize.data_ready[i].clear()
                    Synchronize.result_ready[i].set()

        thread = threading.Thread(target=_worker)
        thread.daemon = True
        thread.start()

    def all_reduce_thread(input):

        if not Synchronize.has_Listener:
            return input

        input_device = input.get_device()
        Synchronize.data_list[input_device] = input
        with torch.cuda.device(input_device):
            Synchronize.result_list[input_device] = type(input)(input.size()).zero_()
        Synchronize.result_ready[input_device].clear()
        Synchronize.data_ready[input_device].set()
        Synchronize.result_ready[input_device].wait()

        return Synchronize.result_list[input_device]

    def forward(ctx, input):
        return Synchronize.all_reduce_thread(input)

    def backward(ctx, gradOutput):
        return Synchronize.all_reduce_thread(gradOutput)


class _sync_batch_norm(Function):
    def __init__(self, momentum, eps):
        super(_sync_batch_norm, self).__init__()
        self.momentum = momentum
        self.eps = eps

    def forward(self, input, running_mean, running_var, weight, bias):
        allreduce_num = Synchronize.device_num

        with torch.cuda.device_of(input):
            mean = input.new().resize_(input.size(1)).zero_()
            var = input.new().resize_(input.size(1)).zero_()
            x_std = input.new().resize_(input.size(1)).zero_()
            x_norm = input.new().resize_as_(input)
            output = input.new().resize_as_(input)

        sync_bn_lib.bn_forward_mean_before_allreduce(input, mean, allreduce_num)
        mean = Synchronize.all_reduce_thread(mean)
        sync_bn_lib.bn_forward_var_before_allreduce(input, mean, var, output, allreduce_num)
        var = Synchronize.all_reduce_thread(var)

        sync_bn_lib.bn_forward_after_allreduce(mean, running_mean, var, running_var, x_norm, x_std, weight, bias, output, self.eps, 1.0 - self.momentum)

        self.save_for_backward(weight, bias)
        self.mean = mean
        self.x_norm = x_norm
        self.x_std = x_std

        return output

    def backward(self, grad_output):
        weight, bias = self.saved_tensors
        allreduce_num = Synchronize.device_num

        with torch.cuda.device_of(grad_output):
            grad_input = grad_output.new().resize_as_(grad_output).zero_()
            grad_weight = grad_output.new().resize_as_(weight).zero_()
            grad_bias = grad_output.new().resize_as_(bias).zero_()
            grad_local_weight = grad_output.new().resize_as_(weight).zero_()
            grad_local_bias = grad_output.new().resize_as_(bias).zero_()

        sync_bn_lib.bn_backward_before_allreduce(grad_output, self.x_norm, self.mean, self.x_std, grad_input, grad_local_weight, grad_local_bias, grad_weight, grad_bias)

        grad_local_weight = Synchronize.all_reduce_thread(grad_local_weight)
        grad_local_bias = Synchronize.all_reduce_thread(grad_local_bias)

        sync_bn_lib.bn_backward_after_allreduce(grad_output, self.x_norm, grad_local_weight, grad_local_bias, weight, self.x_std, grad_input, allreduce_num)

        return grad_input, None, None, grad_weight, grad_bias


def sync_batch_norm(input, running_mean, running_var, weight=None, bias=None, momentum=0.1, eps=1e-5):
    r"""Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _torch_ext.batchnormtrain:

    .. math::

        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    return _sync_batch_norm(momentum, eps)(input, running_mean, running_var, weight, bias)