import torch
import torch.nn as nn

from signbert.model.thirdparty.st_gcn.net.st_gcn import HeadlessModel as STGCN

from IPython import embed

class ArmsExtractor(nn.Module):
    """
    A PyTorch module for extracting and processing arm keypoints using STGCN.

    This module is specifically designed to process sequences of keypoints, focusing on the arms.
    It uses a Spatial Temporal Graph Convolutional Network (STGCN) to extract features from the arm
    keypoints and applies max pooling to these features.

    Attributes:
    stgcn (STGCN): The Spatial Temporal Graph Convolutional Network used for feature extraction.
    """
    def __init__(
        self,
        in_channels,
        hid_dim,
        dropout
    ):
        """
        Initialize the ArmsExtractor module.

        Parameters:
        in_channels (int): The number of input channels (features).
        hid_dim (int): The dimensionality of the hidden layers in STGCN.
        dropout (float): The dropout rate for regularization in STGCN.
        """
        super().__init__()
        self.stgcn = STGCN(
            in_channels=in_channels,
            num_hid=hid_dim,
            graph_args={'layout': 'mmpose_arms'},
            edge_importance_weighting=False,
            dropout=dropout
        )

    def forward(self, x):
        """
        Forward pass for the ArmsExtractor module.

        Parameters:
        x (Tensor): The input tensor containing keypoints data.

        Returns:
        tuple: A tuple of tensors representing processed right and left arm keypoints.
        """
        # Compute the lengths of the sequences (excluding zero-padding)
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # Permute and reshape the input tensor for STGCN
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        # Process the input using STGCN
        x = self.stgcn(x, lens)
        # Extract right and left arm keypoints indices
        rarm = x[:, :, :, (1,3,5)]
        larm = x[:, :, :, (0,2,4)]
        # Apply max pooling to the arm keypoints
        rarm = torch.amax(rarm, dim=3, keepdim=True)
        larm = torch.amax(larm, dim=3, keepdim=True)

        return (rarm, larm)