import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """
    A PyTorch module serving as the head for the Isolated Sign Language 
    Recognition task.

    This module combines features from right and left hand data, applies
    temporal merging, and uses a linear layer as a classifier to predict class
    labels.

    Attributes:
    temporal_merging (nn.Sequential): A sequential model for merging temporal 
    features.
    classifier (nn.Linear): A linear layer for classification.
    """
    def __init__(self, in_channels, num_classes):
        """
        Initialize the Head module.

        Parameters:
        in_channels (int): The number of input channels (features) per hand.
        num_classes (int): The number of classes for the classification task.
        """
        super().__init__()
        # Adjust in_channels to account for concatenation of right and left hand features
        in_channels = in_channels * 2
        # Define the temporal merging layer
        self.temporal_merging = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=1)
        )
        # Define the classification layer
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def forward(self, rhand, lhand):
        """
        Forward pass of the Head module.

        Parameters:
        rhand (Tensor): Input tensor for right hand features.
        lhand (Tensor): Input tensor for left hand features.

        Returns:
        Tensor: The output tensor after classification.
        """
        # Concatenate right and left hand features
        x = torch.concat((rhand, lhand), axis=2)
        # Apply temporal merging to the concatenated features
        x = self.temporal_merging(x) * x
        # Apply max-pooling over the time dimension
        x = F.max_pool1d(x.mT, kernel_size=x.shape[1]).squeeze()
        # Pass through the classifier to obtain final logits
        x = self.classifier(x)

        return x
        