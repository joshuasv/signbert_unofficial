import torch
from torch import nn

from IPython import embed; from sys import exit

class MediapipeHandPooling(nn.Module):
    """
    A PyTorch module for pooling hand keypoints as per MediaPipe's hand tracking configuration.

    This module pools keypoints corresponding to different parts of the hand (palm, thumb,
    index finger, etc.) and is designed to work with MediaPipe's 21 hand keypoints format.

    Attributes:
    PALM_IDXS, THUMB_IDXS, INDEX_IDXS, MIDDLE_IDXS, RING_IDXS, PINKY_IDXS: Tuples containing
    the indices of the keypoints for each part of the hand.
    """
    # Define the indices for different parts of the hand according to MediaPipe's keypoints
    PALM_IDXS = (0, 1, 5, 9, 13, 17)
    THUMB_IDXS = (2, 3, 4)
    INDEX_IDXS = (6, 7, 8)
    MIDDLE_IDXS = (10, 11, 12)
    RING_IDXS = (14, 15, 16)
    PINKY_IDXS = (18, 19, 20)

    def __init__(self, last=False):
        """
        Initialize the pooling module.

        Parameters:
        last (bool): If True, a previous pooling has been made so new pooling is
        applied across the last dimension only. Default is False.
        """
        super().__init__()
        self.last = last

    def forward(self, x):
        """
        Apply pooling operation to hand keypoints.

        Parameters:
        x (Tensor): The input tensor containing keypoints data.

        Returns:
        Tensor: The output tensor after applying pooling.
        """
        if self.last: # If previous pooling has been made
            # Ensure there is only six clusters
            assert x.shape[3] == 6
            return torch.amax(x, 3, keepdim=True)
        else:
            # Ensure that there are 21 hand keypoints
            assert x.shape[3] == 21
            # Apply max pooling to each group of keypoints (palm, thumb, 
            # fingers). Six in total.
            return torch.cat((
                torch.amax(x[:, :, :, MediapipeHandPooling.PALM_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.THUMB_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.INDEX_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.MIDDLE_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.RING_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.PINKY_IDXS], 3, keepdim=True),
            ), dim=3)