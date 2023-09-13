import pickle

import torch
import numpy as np
from torch import nn

from signbert.model.hand_model.rodrigues_batch import rodrigues_batch
from IPython import embed; from sys import exit

class HandAwareModelDecoder(nn.Module):
    c_r_size = 3*3
    c_o_size = 2
    c_s_size = 1
    beta_size = 10
    global_npose_els = 3

    # homo_row = tf.reshape(tf.constant([0.0, 0.0, 0.0, 1.0]), shape=[1, 1, 1, 1, 4])
    homo_row = torch.tensor(np.array([0., 0., 0., 1]), dtype=torch.float32).reshape((1, 1, 1, 1, 4))
    
    # kintree_parent_idxs = tf.constant([0,1,2,0,4,5,0,7,8,0,10,11,0,13,14], dtype=tf.int32)
    kintree_parent_idxs = torch.tensor(np.array([0,1,2,0,4,5,0,7,8,0,10,11,0,13,14]), dtype=torch.int64)


    def __init__(self, num_hid, n_pca_components, mano_model_file):
        super().__init__()
        self.n_pca_components = n_pca_components
        self.theta_size = n_pca_components
        self.mano_model_file = mano_model_file

        # load pre-trained MANO and populate model related variables
        raw_data = self._load_mano_from_disk(self.mano_model_file)
        # hand PCA principal components
        hand_ppal_comps = raw_data['hands_components'][:self.n_pca_components]
        # self.hand_ppal_comps = tf.Variable(self.hand_ppal_comps, trainable=False, dtype=tf.float32, name='hand_ppal_comps')
        self.register_buffer('hand_ppal_comps', torch.tensor(hand_ppal_comps, dtype=torch.float32))
        # mean hand
        mean_hand = raw_data['hands_mean']
        # self.mean_hand = tf.Variable(self.mean_hand, trainable=False, dtype=tf.float32, name='mean_hand')
        self.register_buffer('mean_hand', torch.tensor(mean_hand, dtype=torch.float32))
        # hand template
        hand_template = raw_data['v_template']
        # self.hand_template = tf.Variable(self.hand_template, trainable=False, dtype=tf.float32, name='hand_template')
        self.register_buffer('hand_template', torch.tensor(hand_template, dtype=torch.float32))
        # shape blend function parameters
        shapedirs = raw_data['shapedirs']
        # self.shapedirs = tf.Variable(self.shapedirs, trainable=False, dtype=tf.float32, name='shapedirs')
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32))
        # pose blend function parameters
        posedirs = raw_data['posedirs']
        # self.posedirs = tf.Variable(self.posedirs, trainable=False, dtype=tf.float32, name='posedirs')
        self.register_buffer('posedirs', torch.tensor(posedirs, dtype=torch.float32))
        J_reg = raw_data['J_regressor']
        # self.J_reg = tf.Variable(self.J_reg, trainable=False, dtype=tf.float32, name='J_reg')
        self.register_buffer('J_reg', torch.tensor(J_reg, dtype=torch.float32))
        kintree_table = raw_data['kintree_table']
        # self.kintree_table = tf.Variable(self.kintree_table, trainable=False, dtype=tf.float32, name='kintree_table')
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.float32))
        # LBS weights
        lbs_weights = raw_data['weights']
        # self.lbs_weights = tf.Variable(self.lbs_weights, trainable=False, dtype=tf.float32, name='lbs_weights')
        self.register_buffer('lbs_weights', torch.tensor(lbs_weights, dtype=torch.float32))
        
        # self.hand_model_params = tf.keras.layers.Dense(
        #     units=(self.theta_size +  HandAwareModelDecoder.beta_size + 
        #     HandAwareModelDecoder.c_r_size + HandAwareModelDecoder.c_o_size + 
        #     HandAwareModelDecoder.c_s_size + HandAwareModelDecoder.global_npose_els), name='hand_model_params')
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
        # batch = tf.shape(hand_posed)[0]
        # frames = tf.shape(hand_posed)[1]
        # hand_posed = tf.slice(hand_posed, [0, 0, HandAwareModelDecoder.global_npose_els], [-1, -1, -1])
        # hand_posed = tf.reshape(hand_posed, [batch, frames, -1, 3])
        # rod = rodrigues_batch(hand_posed)
        # rod = tf.reshape(rod, [batch, frames, -1])
        return rod
    
    def _global_rigid_transformation(self, pose, J):
        # batch = tf.shape(pose)[0]
        # frames = tf.shape(pose)[1]
        # n_joints = tf.shape(J)[2]
        pose_shape = pose.shape
        batch = pose_shape[0]
        frames = pose_shape[1]
        n_joints = J.shape[2]

        # function to convert an array to homogeneous coordinates
        to_homo_coords = lambda arr, n_joints: torch.concat((
            arr, 
            HandAwareModelDecoder.homo_row.repeat(batch, frames, n_joints, 1, 1).to(arr.device)
            # tf.repeat(tf.repeat(tf.repeat(HandAwareModelDecoder.homo_row, repeats=batch, axis=0), repeats=frames, axis=1), repeats=n_joints, axis=2)
        ), dim=3)
        # obtain rot matrices using Rodrigues
        # pose = tf.reshape(pose, [batch, frames, -1, 3])
        pose = pose.reshape((batch, frames, -1, 3))
        rod = rodrigues_batch(pose)
        # apply transformation to the first joint (global)
        # _0result = tf.concat([
        #     rod[:, :, 0],
        #     tf.reshape(J[:, :, 0], shape=[batch, frames, 3, 1])
        # ], axis=3)
        # _0result = tf.expand_dims(_0result, axis=2)
        # _0result = to_homo_coords(_0result, 1)
        # _0result = tf.squeeze(_0result, axis=2)
        _0result = torch.concat((
            rod[:, :, 0],
            J[:, :, 0].reshape((batch, frames, 3, 1))
        ), dim=3)
        _0result = _0result.unsqueeze(2)
        _0result = to_homo_coords(_0result, 1)
        _0result = _0result.squeeze(2)

        # subtract children from parent joints(not taking into account global)
        # operand = J[:, :, 1:] - tf.gather(J, indices=HandAwareModelDecoder.kintree_parent_idxs, axis=2)
        # operand = tf.reshape(operand, shape=[batch, frames, n_joints-1, 3, 1])
        operand = J[:, :, 1:] - J[:, :, HandAwareModelDecoder.kintree_parent_idxs]
        operand = operand.reshape((batch, frames, n_joints-1, 3, 1))
        # stack Rodrigues
        # operand = tf.concat([
        #     # tf.gather(rod, indices=HandAwareModelDecoder.kintree_parent_idxs, axis=2),
        #     tf.slice(rod, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1]),
        #     operand
        # ], axis=4)
        operand = torch.concat((
            rod[:, :, 1:],
            operand
        ), dim=4)
        # to homogeneus coords
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
        # apply transformation
        results = torch.stack((
            _0result, _1result, _2result, _3result,  _4result, _5result, 
            _6result, _7result, _8result, _9result,  _10result, _11result, 
            _12result, _13result, _14result, _15result
        ), dim=2)

        # compute transformed mesh, needed to get the extra fingertips keypoints
        extended_J = torch.concat([J, torch.zeros((batch, frames, n_joints, 1), device=J.device)], axis=3)
        mesh = torch.einsum('bfijk,bfik->bfij', results, extended_J)
        mesh = torch.concat((
            torch.zeros((batch, frames, n_joints, 4, 3), device=mesh.device),
            mesh.unsqueeze(4)
            # tf.expand_dims(mesh, axis=4)
        ), dim=4)
        mesh = results - mesh

        # get xyz joint coordinates
        results = results[:, :, :, :3, 3]

        return mesh, results
    
    def forward(self, x):
        # batch = tf.shape(inputs)[0]
        # frames = tf.shape(inputs)[1]
        N, T, _ = x.shape

        # inputs = tf.reduce_mean(inputs, axis=2, keepdims=False)
        params = self.hand_model_params(x)
        # theta = tf.slice(params, [0, 0, 0], [-1, -1, HandAwareModelDecoder.global_npose_els + self.theta_size])
        theta = params[:, :, :HandAwareModelDecoder.global_npose_els + self.theta_size]
        offset = HandAwareModelDecoder.global_npose_els + self.theta_size
        # beta = tf.slice(params, [0, 0, offset], [-1, -1, self.beta_size])
        beta = params[:, :, offset:offset+self.beta_size]
        offset += self.beta_size
        # c_r = tf.slice(params, [0, 0, offset], [-1, -1, HandAwareModelDecoder.c_r_size])
        c_r = params[:, :, offset:offset+HandAwareModelDecoder.c_r_size]
        offset += HandAwareModelDecoder.c_r_size
        # c_o = tf.slice(params, [0, 0, offset], [-1, -1, HandAwareModelDecoder.c_o_size])
        c_o = params[:, :, offset:offset+HandAwareModelDecoder.c_o_size]
        offset += HandAwareModelDecoder.c_o_size
        # c_s = tf.slice(params, [0, 0, offset], [-1, -1, HandAwareModelDecoder.c_s_size])
        c_s = params[:, :, offset:offset+HandAwareModelDecoder.c_s_size]

        # pose hand
        # ppal_comps_pose = tf.slice(theta, [0, 0, HandAwareModelDecoder.global_npose_els], [-1, -1, self.n_pca_components])
        ppal_comps_pose = theta[:, :, HandAwareModelDecoder.global_npose_els:HandAwareModelDecoder.global_npose_els+self.n_pca_components]
        # hand_posed = tf.matmul(ppal_comps_pose, self.hand_ppal_comps)
        hand_posed = torch.matmul(ppal_comps_pose, self.hand_ppal_comps)
        hand_posed = hand_posed + self.mean_hand

        # concatenate global pose
        # hand_posed = tf.concat(
        #     [tf.slice(theta, [0, 0, 0], [-1, -1, HandAwareModelDecoder.global_npose_els]), hand_posed],
        #     axis=-1
        # )
        hand_posed = torch.concat((
            theta[:, :, :HandAwareModelDecoder.global_npose_els],
            hand_posed
        ), dim=-1)

        # shape blend function (Bs in the paper)
        # hand_shaped = tf.einsum('ijk,btk->btij', self.shapedirs, beta)
        hand_shaped = torch.einsum('ijk,btk->btij', self.shapedirs, beta)
        hand_shaped = hand_shaped + self.hand_template

        # compute joints location as a function of the shape (J(beta) in the paper)
        # J = tf.einsum('ij,btjc->btic', self.J_reg, hand_shaped)
        J = torch.einsum('ij,btjc->btic', self.J_reg, hand_shaped)

        # from vector to matrix rotation using Rodrigues formula
        rod = self._lrotmin(hand_posed)
        # apply rotation
        # rot = tf.einsum('ijk,bfk->bfij', self.posedirs, rod)
        rot = torch.einsum('ijk,bfk->bfij', self.posedirs, rod)
        # apply to hand
        curr_hand = hand_shaped + rot

        # compute hand mesh transformation and 3d joint coordinates
        A, J3d = self._global_rigid_transformation(pose=hand_posed, J=J)

        # compute hand mesh final 3d position
        # hand_mesh_3d = tf.einsum('bfjik,cj->bfcik', A, self.lbs_weights)
        # n_vertices = tf.shape(curr_hand)[2]
        # curr_hand = tf.concat([
        #     curr_hand,
        #     tf.ones([batch, frames, n_vertices, 1])
        # ], axis=3)
        # hand_mesh_3d = tf.einsum('bfvij,bfvj->bfvi', hand_mesh_3d, curr_hand)
        # hand_mesh_3d = tf.slice(hand_mesh_3d, [0, 0, 0, 0], [-1, -1, -1, 3])
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

        # ortographic projection of 3d joints
        # c_r = tf.reshape(c_r, shape=[batch, frames, 3, 3])
        # c_s = tf.expand_dims(c_s, axis=2)
        # c_o = tf.expand_dims(c_o, axis=2)
        # J2d = (c_s * orthoproj(tf.matmul(J3d_reordered, c_r))) + c_o
        c_r = c_r.reshape((N, T, 3, 3))
        c_s = c_s.unsqueeze(2)
        c_o = c_o.unsqueeze(2)
        J2d = torch.matmul(J3d_reordered, c_r)
        # Orthographic projection
        J2d = J2d[...,:2]
        J2d = (c_s * J2d) + c_o

        return J2d, theta, beta