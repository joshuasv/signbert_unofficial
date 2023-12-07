from typing import Optional
import torch
from torch import Tensor
from torch.nn.modules.pooling import _MaxPoolNd
from torch.nn.common_types import _size_any_t
from torch.nn import functional as F
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

class _MaskedMaxPool(_MaxPoolNd):
    
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
                 padding: _size_any_t = 0, dilation: _size_any_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )
    
    def forward(self, input: Tensor, lengths: Tensor = None) -> Tensor:
        cls_name = self.__class__.__name__
        
        if lengths is not None:
            mask = lengths_to_mask(lengths, max_len=input.shape[2], dtype=torch.bool)
            mask = torch.logical_not(mask)
            if '1d' in cls_name:
                mask = mask.unsqueeze(1).expand(input.shape)
            elif '2d' in cls_name:
                mask = mask.unsqueeze(1).unsqueeze(-1).expand(input.shape)
            else:
                raise NotImplementedError(f'Not implemented {cls_name=}.')
        
            or_values = input[mask].detach().cpu()
            input.masked_fill_(mask, torch.tensor(float("-inf"), dtype=torch.float32))

        if '1d' in cls_name:
            pool_fn = F.max_pool1d
        elif '2d' in cls_name:
            pool_fn = F.max_pool2d
        else:
            raise NotImplementedError(f'Not implemented {cls_name=}.')

        input = pool_fn(input, self.kernel_size, self.stride,
            self.padding, self.dilation, ceil_mode=self.ceil_mode,
        return_indices=self.return_indices)

        if mask.shape == input.shape:
            input[mask] = or_values.to(input.device)

        print("ENTERNETENRNERNERNENRENR1")
        assert input.min() != float("-inf")

        return input

class MaskedMaxPool2d(_MaskedMaxPool):

    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
                 padding: _size_any_t = 0, dilation: _size_any_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )