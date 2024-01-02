import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed


class Head(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        in_channels = in_channels * 2
        self.temporal_merging = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def forward(self, rhand, lhand):
        x = torch.concat((rhand, lhand), axis=2)
        x = self.temporal_merging(x) * x
        x = F.max_pool1d(x.mT, kernel_size=x.shape[1]).squeeze()
        x = self.classifier(x)

        return x
        