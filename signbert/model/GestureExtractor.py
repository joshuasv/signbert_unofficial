from torch import nn

from signbert.model.st_gcn.net.st_gcn import HeadlessModel as STGCN
from IPython import embed; from sys import exit

class GestureExtractor(nn.Module):

    def __init__(self, in_channels, num_hid):
        super().__init__()
        # TODO; STGCN customizable number of layers and dimensions
        self.stgcn = STGCN(in_channels, num_hid, {'layout': 'mediapipe_hand'}, edge_importance_weighting=False)

    def forward(self, x):
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # STGCN expects inputs with shape (N, C, T, V, M)
        x = x.unsqueeze(-1).permute(0, 3, 1, 2, 4)
        x = self.stgcn(x, lens)
        
        return x

if __name__ == '__main__':
    import torch
    import numpy as np
    from signbert.data_modules.MaskKeypointDataset import MaskKeypointDataset

    dataset = MaskKeypointDataset(
        npy_fpath='/home/temporal2/jsoutelo/datasets/HANDS17/preprocess/X_train.npy',
        R=0.2,
        m=5,
        K=6
    )
    ge = GestureExtractor(in_channels=2)

    # Create test batch
    seq, _ = dataset[0]
    seq = torch.tensor(np.stack((seq, seq)).astype(np.float32))
    # Forward
    out = ge(seq)
    print(f'{out[0].shape=}')
    embed(); exit()