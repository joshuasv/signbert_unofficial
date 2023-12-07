import torch
from torch import nn

from IPython import embed; from sys import exit

class MediapipeHandPooling(nn.Module):

    PALM_IDXS = (0, 1, 5, 9, 13, 17)
    THUMB_IDXS = (2, 3, 4)
    INDEX_IDXS = (6, 7, 8)
    MIDDLE_IDXS = (10, 11, 12)
    RING_IDXS = (14, 15, 16)
    PINKY_IDXS = (18, 19, 20)

    def __init__(self, last=False):
        super().__init__()
        self.last = last

    def forward(self, x):
        if self.last:
            assert x.shape[3] == 6
            return torch.amax(x, 3, keepdim=True)
        else:
            assert x.shape[3] == 21
            return torch.cat((
                torch.amax(x[:, :, :, MediapipeHandPooling.PALM_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.THUMB_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.INDEX_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.MIDDLE_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.RING_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.PINKY_IDXS], 3, keepdim=True),
            ), dim=3)