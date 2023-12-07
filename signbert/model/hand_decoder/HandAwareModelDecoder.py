import torch
from torch import nn

from signbert.mano.manolayer import ManoLayer
from IPython import embed; from sys import exit


class HandAwareModelDecoder(nn.Module):
    C_R_SIZE = 3 * 3
    C_O_SIZE = 2
    C_S_SIZE = 1
    BETA_SIZE = 10
    GLOBAL_NPOSE_ELS = 3

    def __init__(
        self,
        in_features,
        n_pca_components,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_pca_components = n_pca_components

        self.params = nn.Linear(
            in_features=in_features,
            out_features=(
                n_pca_components +
                HandAwareModelDecoder.BETA_SIZE + 
                HandAwareModelDecoder.C_R_SIZE + 
                HandAwareModelDecoder.C_O_SIZE + 
                HandAwareModelDecoder.C_S_SIZE + 
                HandAwareModelDecoder.GLOBAL_NPOSE_ELS
            )
        )
        self.mano = ManoLayer(
            center_idx=None, # TODO; wrist?
            flat_hand_mean=False,
            ncomps=n_pca_components,
            use_pca=n_pca_components != 45
        )

    def forward(self, x):
        N, L = x.shape[:2]
        params = self.params(x)
        offset = HandAwareModelDecoder.GLOBAL_NPOSE_ELS + self.n_pca_components
        theta = params[:, :, :offset]
        beta = params[:, :, offset:offset + self.BETA_SIZE]
        offset += self.BETA_SIZE
        c_r = params[:, :, offset:offset + HandAwareModelDecoder.C_R_SIZE]
        offset += HandAwareModelDecoder.C_R_SIZE
        c_o = params[:, :, offset:offset + HandAwareModelDecoder.C_O_SIZE]
        offset += HandAwareModelDecoder.C_O_SIZE
        c_s = params[:, :, offset:offset + HandAwareModelDecoder.C_S_SIZE]

        verts = []
        joints = []
        for seq_idx in range(x.shape[1]):
            v, j, _ = self.mano(th_pose_coeffs=theta[:, seq_idx], th_betas=beta[:, seq_idx])
            verts.append(v)
            joints.append(j)
        verts = torch.stack(verts, dim=1)
        joints = torch.stack(joints, dim=1)
        # TODO; handle the sequence length dimension
        # verts, joints, center_joint = self.mano(th_pose_coeffs=theta, th_betas=beta)

        # Orthographic projection
        # sources: https://sites.ecse.rpi.edu/~qji/CV/perspective_geometry2.pdf
        joints_2d = (c_s.unsqueeze(2) * torch.matmul(joints, c_r.view(N, L, 3, 3))[...,:2]) + c_o.unsqueeze(2)

        return (joints_2d, theta, beta, verts, c_r, c_s, c_o)
        
if __name__ == '__main__':

    HandAwareModelDecoder()