import pickle

import torch
import numpy as np
from torch import nn

from signbert.model.hand_model.rodrigues_batch import rodrigues_batch


class HandAwareModelDecoder(nn.Module):
    c_r_size = 3*3
    c_o_size = 2
    c_s_size = 1
    beta_size = 10
    global_npose_els = 3
    homo_row = torch.tensor(np.array([0., 0., 0., 1]), dtype=torch.float32).reshape((1, 1, 1, 1, 4))
    kintree_parent_idxs = torch.tensor(np.array([0,1,2,0,4,5,0,7,8,0,10,11,0,13,14]), dtype=torch.int64)

    def __init__(self, num_hid, n_pca_components, mano_model_file):
        super().__init__()
        self.n_pca_components = n_pca_components
        self.theta_size = n_pca_components
        self.mano_model_file = mano_model_file

        # Load pre-trained MANO and populate model related variables
        raw_data = self._load_mano_from_disk(self.mano_model_file)
        # Hand PCA principal components
        hand_ppal_comps = raw_data['hands_components'][:self.n_pca_components]
        self.register_buffer('hand_ppal_comps', torch.tensor(hand_ppal_comps, dtype=torch.float32))
        # Mean hand
        mean_hand = raw_data['hands_mean']
        self.register_buffer('mean_hand', torch.tensor(mean_hand, dtype=torch.float32))
        # Hand template
        hand_template = raw_data['v_template']
        self.register_buffer('hand_template', torch.tensor(hand_template, dtype=torch.float32))
        # Shape blend function parameters
        shapedirs = raw_data['shapedirs']
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32))
        # Pose blend function parameters
        posedirs = raw_data['posedirs']
        self.register_buffer('posedirs', torch.tensor(posedirs, dtype=torch.float32))
        J_reg = raw_data['J_regressor']
        self.register_buffer('J_reg', torch.tensor(J_reg, dtype=torch.float32))
        kintree_table = raw_data['kintree_table']
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.float32))
        # LBS weights
        lbs_weights = raw_data['weights']
        self.register_buffer('lbs_weights', torch.tensor(lbs_weights, dtype=torch.float32))
        self.hand_model_params = nn.Linear(
            in_features=num_hid*21, # TODO; remove hardcoded values
            out_features=(
                self.theta_size +  HandAwareModelDecoder.beta_size + 
                HandAwareModelDecoder.c_r_size + 
                HandAwareModelDecoder.c_o_size + 
                HandAwareModelDecoder.c_s_size + 
                HandAwareModelDecoder.global_npose_els
            )
        )

    def _load_mano_from_disk(self, fname):
            with open(fname, 'rb') as fid:
                return pickle.load(fid)

    def _lrotmin(self, hand_posed):
        B, F, _ = hand_posed.shape
        hand_posed = hand_posed[:, :, HandAwareModelDecoder.global_npose_els:]
        hand_posed = hand_posed.reshape((B, F, -1, 3))
        rod = rodrigues_batch(hand_posed)
        rod = rod.reshape((B, F, -1))

        return rod
    
    def _global_rigid_transformation(self, pose, J):
        pose_shape = pose.shape
        batch = pose_shape[0]
        frames = pose_shape[1]
        n_joints = J.shape[2]
        # Function to convert an array to homogeneous coordinates
        to_homo_coords = lambda arr, n_joints: torch.concat((
            arr, 
            HandAwareModelDecoder.homo_row.repeat(batch, frames, n_joints, 1, 1).to(arr.device)
        ), dim=3)
        # Obtain rot matrices using Rodrigues
        pose = pose.reshape((batch, frames, -1, 3))
        rod = rodrigues_batch(pose)
        # Apply transformation to the first joint (global)
        _0result = torch.concat((
            rod[:, :, 0],
            J[:, :, 0].reshape((batch, frames, 3, 1))
        ), dim=3)
        _0result = _0result.unsqueeze(2)
        _0result = to_homo_coords(_0result, 1)
        _0result = _0result.squeeze(2)
        # Subtract children from parent joints(not taking into account global)
        operand = J[:, :, 1:] - J[:, :, HandAwareModelDecoder.kintree_parent_idxs]
        operand = operand.reshape((batch, frames, n_joints-1, 3, 1))
        # Stack Rodrigues
        operand = torch.concat((
            rod[:, :, 1:],
            operand
        ), dim=4)
        # To homogeneus coords
        operand = to_homo_coords(operand, n_joints-1)
        # TODO: need a better way
        _1result = torch.matmul(_0result, operand[:, :, 0])
        _2result = torch.matmul(_1result, operand[:, :, 1])
        _3result = torch.matmul(_2result, operand[:, :, 2])
        _4result = torch.matmul(_0result, operand[:, :, 3])
        _5result = torch.matmul(_4result, operand[:, :, 4])
        _6result = torch.matmul(_5result, operand[:, :, 5])
        _7result = torch.matmul(_0result, operand[:, :, 6])
        _8result = torch.matmul(_7result, operand[:, :, 7])
        _9result = torch.matmul(_8result, operand[:, :, 8])
        _10result = torch.matmul(_0result, operand[:, :, 9])
        _11result = torch.matmul(_10result, operand[:, :, 10])
        _12result = torch.matmul(_11result, operand[:, :, 11])
        _13result = torch.matmul(_0result, operand[:, :, 12])
        _14result = torch.matmul(_13result, operand[:, :, 13])
        _15result = torch.matmul(_14result, operand[:, :, 14])
        # Apply transformation
        results = torch.stack((
            _0result, _1result, _2result, _3result,  _4result, _5result, 
            _6result, _7result, _8result, _9result,  _10result, _11result, 
            _12result, _13result, _14result, _15result
        ), dim=2)
        # Compute transformed mesh, needed to get the extra fingertips keypoints
        extended_J = torch.concat([J, torch.zeros((batch, frames, n_joints, 1), device=J.device)], axis=3)
        mesh = torch.einsum('bfijk,bfik->bfij', results, extended_J)
        mesh = torch.concat((
            torch.zeros((batch, frames, n_joints, 4, 3), device=mesh.device),
            mesh.unsqueeze(4)
        ), dim=4)
        mesh = results - mesh

        # Get xyz joint coordinates
        results = results[:, :, :, :3, 3]

        return mesh, results
    
    def forward(self, x):
        N, T, _ = x.shape

        params = self.hand_model_params(x)
        theta = params[:, :, :HandAwareModelDecoder.global_npose_els + self.theta_size]
        offset = HandAwareModelDecoder.global_npose_els + self.theta_size
        beta = params[:, :, offset:offset+self.beta_size]
        offset += self.beta_size
        c_r = params[:, :, offset:offset+HandAwareModelDecoder.c_r_size]
        offset += HandAwareModelDecoder.c_r_size
        c_o = params[:, :, offset:offset+HandAwareModelDecoder.c_o_size]
        offset += HandAwareModelDecoder.c_o_size
        c_s = params[:, :, offset:offset+HandAwareModelDecoder.c_s_size]
        # Pose hand
        ppal_comps_pose = theta[:, :, HandAwareModelDecoder.global_npose_els:HandAwareModelDecoder.global_npose_els+self.n_pca_components]
        hand_posed = torch.matmul(ppal_comps_pose, self.hand_ppal_comps)
        hand_posed = hand_posed + self.mean_hand
        # Concatenate global pose
        hand_posed = torch.concat((
            theta[:, :, :HandAwareModelDecoder.global_npose_els],
            hand_posed
        ), dim=-1)
        # Shape blend function (Bs in the paper)
        hand_shaped = torch.einsum('ijk,btk->btij', self.shapedirs, beta)
        hand_shaped = hand_shaped + self.hand_template
        # Compute joints location as a function of the shape (J(beta) in the paper)
        J = torch.einsum('ij,btjc->btic', self.J_reg, hand_shaped)
        # From vector to matrix rotation using Rodrigues formula
        rod = self._lrotmin(hand_posed)
        # Apply rotation
        rot = torch.einsum('ijk,bfk->bfij', self.posedirs, rod)
        # Apply to hand
        curr_hand = hand_shaped + rot
        # Compute hand mesh transformation and 3d joint coordinates
        A, J3d = self._global_rigid_transformation(pose=hand_posed, J=J)
        # Compute hand mesh final 3d position
        hand_mesh_3d = torch.einsum('bfjik,cj->bfcik', A, self.lbs_weights)
        n_vertices = curr_hand.shape[2]
        curr_hand = torch.concat((
            curr_hand,
            torch.ones((N, T, n_vertices, 1), device=curr_hand.device)
        ), dim=3)
        hand_mesh_3d = torch.einsum('bfvij,bfvj->bfvi', hand_mesh_3d, curr_hand)
        hand_mesh_3d = hand_mesh_3d[...,:3]
        # Reorder joints and append extra fingertips so it matches mediapipe
        J3d_reordered = torch.stack((
            J3d[:, :, 0],
            J3d[:, :, 13],
            J3d[:, :, 14],
            J3d[:, :, 15],
            hand_mesh_3d[:, :, 743],
            J3d[:, :, 1],
            J3d[:, :, 2],
            J3d[:, :, 3],
            hand_mesh_3d[:, :, 333],
            J3d[:, :, 4],
            J3d[:, :, 5],
            J3d[:, :, 6],
            hand_mesh_3d[:, :, 443],
            J3d[:, :, 10],
            J3d[:, :, 11],
            J3d[:, :, 12],
            hand_mesh_3d[:, :, 555],
            J3d[:, :, 7],
            J3d[:, :, 8],
            J3d[:, :, 9],
            hand_mesh_3d[:, :, 678],
        ), axis=2)
        # Ortographic projection of 3d joints
        c_r = c_r.reshape((N, T, 3, 3))
        c_s = c_s.unsqueeze(2)
        c_o = c_o.unsqueeze(2)
        J2d = torch.matmul(J3d_reordered, c_r)
        J2d = J2d[...,:2]
        J2d = (c_s * J2d) + c_o
        # Ortographic projection of 3d hand mesh
        hand_mesh_2d = torch.matmul(hand_mesh_3d, c_r)
        hand_mesh_2d = hand_mesh_2d[...,:2]
        hand_mesh_2d = (c_s * hand_mesh_2d) + c_o

        return J2d, theta, beta, hand_mesh_2d