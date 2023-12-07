from torch import nn
import torch.nn.functional as F

from signbert.model.spat_temp_st_gcn.nets.st_gcn_multi_frame import HeadlessModel as STGCN
from IPython import embed; from sys import exit

class Options:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class GestureExtractor(nn.Module):

    def __init__(
            self,
            in_channels,
            inter_channels,
            fc_unit,
            layout,
            strategy,
            pad,
        ):
        super().__init__()
        kwargs = dict(
            in_channels=in_channels,
            inter_channels=inter_channels,
            fc_unit=fc_unit,
            layout=layout,
            strategy=strategy,
            pad=pad,
        )
        self.opt = Options(**kwargs)
        self.stgcn = STGCN(self.opt)
        self.kernel = 3
        self.to_pad = int((self.kernel - 1) / 2)

    def forward(self, x):
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # Pad so final output has same dimension as input
        # p4d = (0, 0, 0, 0, self.to_pad, self.to_pad)
        # x = F.pad(x, p4d, mode='replicate')
        # # Sliding window across temporal dimension
        # x = x.unfold(dimension=1, size=self.kernel, step=1)
        # N, T, V, C, S_t = x.shape
        N, T, V, C = x.shape
        # # STGCN expects inputs with shape (N, C, T, V, M)
        # x = x.reshape(N*T, V, C, S_t, 1).permute(0, 2, 3, 1, 4)
        x = x.unsqueeze(-1).permute(0, 3, 1, 2, 4)
        x = self.stgcn(x, lens)
        x = x[:, :, 1] # Grab the middle frame
        _, C, V = x.shape
        x = x.view(N, T, C, V).permute(0, 2, 1, 3).unsqueeze(-1)

        return x

if __name__ == '__main__':
    import torch
    import numpy as np
    import torch.nn.functional as F
    from signbert.data_modules.MaskKeypointDataset import MaskKeypointDataset

    dataset = MaskKeypointDataset(
        idxs_fpath='/home/temporal2/jsoutelo/datasets/HANDS17/preprocess/idxs.npy',
        npy_fpath='/home/temporal2/jsoutelo/datasets/HANDS17/preprocess/X_train.npy',
        R=0.2,
        m=5,
        K=6
    )
    ge = GestureExtractor(
        in_channels=2,
        inter_channels=[256, 256],
        fc_unit=512,
        layout='mediapipe_hand',
        strategy='spatial',
        pad=1,
    )
    # Create test batch
    _, seq, _, _, _ = dataset[0]
    # seq = torch.tensor(np.stack((seq, seq)).astype(np.float32))
    # seq = torch.tensor(seq, dtype=torch.float32)
    # kernel = 3
    # to_pad = int((kernel - 1) / 2)
    # p3d = (0, 0, 0, 0, to_pad, to_pad)
    # seq = F.pad(seq, p3d, 'constant', 0)
    # # seqlen = ge.opt.pad*2+1
    # # add pad to sequence so it has same output
    # seq_u = seq.unfold(dimension=0, size=seqlen, step=1)
    # seq_u = seq_u.permute(0, 3, 1, 2)
    # # Forward
    # out = ge(seq_u)
    # print(f'{out[0].shape=}')
    # embed(); exit()