from torch import nn

from signbert.model.st_gcn.net.st_gcn import HeadlessModel as STGCN
from signbert.model.MediapipeHandPooling import MediapipeHandPooling
from IPython import embed; from sys import exit

class GestureExtractor(nn.Module):

    def __init__(self, in_channels, num_hid, dropout=0):
        super().__init__()
        # TODO; STGCN customizable number of layers and dimensions
        self.stgcn1 = STGCN(
            in_channels, 
            num_hid, 
            {'layout': 'mediapipe_hand'}, 
            edge_importance_weighting=False,
            dropout=dropout
        )
        # self.stgcn2 = STGCN(
        #     num_hid, 
        #     num_hid, 
        #     {'layout': 'mediapipe_hand'}, 
        #     edge_importance_weighting=False
        # )

    def forward(self, x):
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # STGCN expects inputs with shape (N, C, T, V, M)
        x = x.unsqueeze(-1).permute(0, 3, 1, 2, 4)
        x = self.stgcn1(x, lens)
        # x = self.hand_pool_six_cluster(x)
        # x = self.stgcn2(x, lens)
        # x = self.hand_pool_last(x)
        
        return x

if __name__ == '__main__':
    import torch
    import numpy as np
    from signbert.data_modules.MaskKeypointDataset import MaskKeypointDataset

    dataset = MaskKeypointDataset(
        idxs_fpath='/home/temporal2/jsoutelo/datasets/HANDS17/preprocess/idxs.npy',
        npy_fpath='/home/temporal2/jsoutelo/datasets/HANDS17/preprocess/X_train.npy',
        R=0.2,
        m=5,
        K=6
    )
    ge = GestureExtractor(in_channels=2, num_hid=256)

    # Create test batch
    _, seq, _, _, _ = dataset[0]
    seq = torch.tensor(np.stack((seq, seq)).astype(np.float32))
    # Forward
    out = ge(seq)
    print(f'{out[0].shape=}')
    embed(); exit()