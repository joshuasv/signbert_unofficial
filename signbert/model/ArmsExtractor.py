import torch
import torch.nn as nn

from signbert.model.thirdparty.st_gcn.net.st_gcn import HeadlessModel as STGCN

from IPython import embed

class ArmsExtractor(nn.Module):

    def __init__(
        self,
        in_channels,
        hid_dim,
        dropout
    ):
        super().__init__()
        self.stgcn = STGCN(
            in_channels=in_channels,
            num_hid=hid_dim,
            graph_args={'layout': 'mmpose_arms'},
            edge_importance_weighting=False,
            dropout=dropout
        )

    def forward(self, x):
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        x = self.stgcn(x, lens)
        rarm = x[:, :, :, (1,3,5)]
        larm = x[:, :, :, (0,2,4)]
        rarm = torch.amax(rarm, dim=3, keepdim=True)
        larm = torch.amax(larm, dim=3, keepdim=True)

        return (rarm, larm)