# Small modification from: https://gist.github.com/ilya16/c622461000480e66ae906dd9dbe8ea26
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from IPython import embed

def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.
    
    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397
    
    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype)\
    .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask

# Masked Batch Normalization

def masked_batch_norm(input: Tensor, mask: Tensor, weight: Optional[Tensor], bias: Optional[Tensor],
                      running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool,
                      momentum: float, eps: float = 1e-5) -> Tensor:
    r"""Applies Masked Batch Normalization for each channel in each data sample in a batch.

    See :class:`~MaskedBatchNorm1d`, :class:`~MaskedBatchNorm2d`, :class:`~MaskedBatchNorm3d` for details.
    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when training=False')
    num_dims = len(input.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims
    if training:
        num_elements = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / num_elements  # (C,)
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements  # (C,)
        # var = (((input - mean.as_strided((1,mean.shape[0])+(1,)*num_dims, (1,)*len(input.shape))) * mask) ** 2).sum(_dims) / num_elements  # (C,)

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = running_mean, running_var

    # mean = mean.as_strided((1,mean.shape[0])+(1,)*num_dims, (1,)*len(input.shape))
    # var = var.as_strided((1,var.shape[0])+(1,)*num_dims, (1,)*len(input.shape))
    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)
    # input = (input - mean) / torch.sqrt(var + eps)
    if weight is not None and bias is not None:
        # weight = mean.as_strided((1,weight.shape[0])+(1,)*num_dims, (1,)*len(input.shape))
        # bias = var.as_strided((1,bias.shape[0])+(1,)*num_dims, (1,)*len(input.shape))
        out = out * weight[_slice] + bias[_slice]
        # input = input * weight + bias

    return out


class _MaskedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MaskedBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    # def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
    def forward(self, input: Tensor, lengths: Tensor = None) -> Tensor:
        self._check_input_dim(input)
        # if mask is not None:
        if lengths is not None:
            cls_name = self.__class__.__name__
            # mask = lengths_to_mask(lengths, max_len=input.shape[2], dtype=input.dtype)
            mask = lengths_to_mask(lengths, max_len=input.shape[2], dtype=torch.bool)
            if '1d' in cls_name:
                mask = mask.unsqueeze(1).expand(input.shape)
            elif '2d' in cls_name:
                mask = mask.unsqueeze(1).unsqueeze(-1).expand(input.shape)
            else:
                raise NotImplementedError(f'Not implemented {cls_name=}.')
            
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if mask is None:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps
            )
        else:
            return masked_batch_norm(
                input, mask, self.weight, self.bias,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                bn_training, exponential_average_factor, self.eps
            )


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.BatchNorm1d` for details.

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super(MaskedBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)


class MaskedBatchNorm2d(torch.nn.BatchNorm2d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 4D input
    (a mini-batch of 2D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.BatchNorm2d` for details.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Mask: :math:`(N, 1, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super(MaskedBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)


class MaskedBatchNorm3d(torch.nn.BatchNorm3d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 5D input
    (a mini-batch of 3D inputs with additional channel dimension).

    See documentation of :class:`~torch.nn.BatchNorm3d` for details.

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Mask: :math:`(N, 1, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super(MaskedBatchNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)