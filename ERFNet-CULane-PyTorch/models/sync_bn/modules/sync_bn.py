import torch
from torch.autograd import Variable
from ..functions import sync_batch_norm
from ..functions import Synchronize

__all__ = ['SyncBatchNorm2d', 'Synchronize', 'convert_bn']


class SyncBatchNorm2d(torch.nn.BatchNorm2d):
    r"""Synchronized Batch Normalization 2d
    Please use compatible :class:`torch_ext.parallel.SelfDataParallel` and :class:`torch_ext.nn`

    Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable
            affine parameters. Default: True

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> m = torch_ext.nn.BatchNorm2d(100).cuda()
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45)).cuda()
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        if isinstance(input, Variable):
            self._check_input_dim(input)
            if self.training and Synchronize.device_num > 1:
                B, C, H, W = input.size()
                rm = Variable(self.running_mean, requires_grad=False)
                rv = Variable(self.running_var, requires_grad=False)

                output = sync_batch_norm(input.view(B, C, -1).contiguous(), rm, rv, self.weight, self.bias, self.momentum, self.eps)

                self.running_mean = rm.data
                self.running_var = rv.data

                return output.view(B, C, H, W)
            else:
                return super(SyncBatchNorm2d, self).forward(input)
        else:
            raise RuntimeError('unknown input type')


def convert_bn(model, memo=None, bn_type=SyncBatchNorm2d):
    bn_list = []
    if memo is None:
        memo = set()
    if model not in memo:
        memo.add(model)
        for name, module in model._modules.items():
            if module is None:
                continue
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_list.append(name)
            convert_bn(module, memo, bn_type)
        for name in bn_list:
            m = model._modules[name]
            temp = bn_type(m.num_features)
            temp.__dict__.update(m.__dict__)
            model._modules[name] = temp
