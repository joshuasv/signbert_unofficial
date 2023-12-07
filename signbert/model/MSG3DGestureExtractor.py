import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from signbert.model.MS_G3D.model.msg3d import HeadlessModel as MSG3D
from signbert.model.MediapipeHandPooling import MediapipeHandPooling
from signbert.model.st_gcn.net.st_gcn import HeadlessModel as STGCN
from torch.nn.functional import dropout
from IPython import embed

num_node = 21
self_link = [(i, i) for i in range(num_node)]
inward = [
    (0, 1), (0, 5), (0, 17), # Wrist
    (5, 9), (9, 13), (13, 17), # Palm
    (1, 2), (2, 3), (3, 4), # Thumb
    (5, 6), (6, 7), (7, 8), # Index
    (9, 10), (10, 11), (11, 12), # Middle
    (13, 14), (14, 15), (15, 16), # Ring
    (17, 18), (18, 19), (19, 20) # Pinky
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Hands17Graph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = self.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = self.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)

    def get_adjacency_matrix(self, edges, num_nodes):
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for edge in edges:
            A[edge] = 1.
        return A
    
class Hands17GraphCluster:
    def __init__(self, *args, **kwargs):
        self.num_nodes = 6
        self.A_binary = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.float32)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class GestureExtractor(nn.Module):

    def __init__(
            self,
            num_point,
            num_gcn_scales,
            num_g3d_scales,
            hid_dim,
            in_channels,
            do_cluster,
            msg_3d_dropout=0.0,
            st_gcn_dropout=0.0,
            dropout=0.0,
            relu_between=False,
        ):
        super().__init__()
        self.do_cluster = do_cluster
        self.relu_between = relu_between

        self.model = MSG3D(
            num_point,
            num_gcn_scales,
            num_g3d_scales,
            Hands17Graph(),
            hid_dim,
            msg_3d_dropout,
            in_channels,
        )
        
        if do_cluster:
            self.maxpool1 = MediapipeHandPooling(last=False)
            self.stgcn = STGCN(
                in_channels=hid_dim[-1],
                num_hid=hid_dim[-1],
                graph_args={'layout': 'mediapipe_six_hand_cluster'},
                edge_importance_weighting=False,
                dropout=st_gcn_dropout
            )
            self.maxpool2 = MediapipeHandPooling(last=True)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # MSG3D expects data in (N, C, T, V, M) format
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        x = self.model(x, lens)
        
        if self.do_cluster:
            x = self.maxpool1(x)
            if self.relu_between:
                x = F.relu(x)
            x = x.unsqueeze(-1)
            x = self.stgcn(x, lens)
            x = x.squeeze(-1)
            x = self.maxpool2(x)
            if self.relu_between:
                x = F.relu(x)            
            x = x.unsqueeze(-1) # Add M dimension
        else:
            x = x.unsqueeze(-1) # Add M dimension

        x = self.dropout(x)
        return x

if __name__ == '__main__':
    from signbert.data_modules.HANDS17DataModule import HANDS17DataModule
    # Create data module
    hands17 = HANDS17DataModule(
        batch_size=32, 
        normalize=True
    )
    hands17.setup()
    dl = hands17.train_dataloader()
    batch = next(iter(dl))
    # Create gesture extractor
    # ge = GestureExtractor(
    #     num_point=21,
    #     num_gcn_scales=1,
    #     num_g3d_scales=1,
    #     in_channels=2,
    # )
    # ge(batch[2])