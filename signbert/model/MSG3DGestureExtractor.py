import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from signbert.model.thirdparty.MS_G3D.model.msg3d import HeadlessModel as MSG3D
from signbert.model.MediapipeHandPooling import MediapipeHandPooling
from signbert.model.thirdparty.st_gcn.net.st_gcn import HeadlessModel as STGCN
from torch.nn.functional import dropout
from IPython import embed


class Hands17Graph:
    """
    A class to represent the graph structure of a hand with 21 keypoints.

    This class creates a graph representation of a hand, with nodes representing keypoints 
    and edges representing the connections between these keypoints. 

    Attributes:
    num_nodes (int): The number of nodes (keypoints) in the graph.
    edges (list): A list of edges representing connections between keypoints.
    self_loops (list): A list of self-loops for each node in the graph.
    A_binary (ndarray): The binary adjacency matrix of the graph without self-loops.
    A_binary_with_I (ndarray): The binary adjacency matrix of the graph with self-loops.
    """
    def __init__(self, *args, **kwargs):
        num_node = 21 # The number of keypoints in a hand
        self.num_nodes = num_node
        # Define the edges of the graph based on hand keypoints connectivity
        inward = [
            (0, 1), (0, 5), (0, 17), # Wrist
            (5, 9), (9, 13), (13, 17), # Palm
            (1, 2), (2, 3), (3, 4), # Thumb
            (5, 6), (6, 7), (7, 8), # Index
            (9, 10), (10, 11), (11, 12), # Middle
            (13, 14), (14, 15), (15, 16), # Ring
            (17, 18), (18, 19), (19, 20) # Pinky
        ]
        # Outward edges are just inward edges reversed
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward
        # Save the edges and self-loops
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        # Generate binary adjacency matrices
        self.A_binary = self.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = self.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)

    def get_adjacency_matrix(self, edges, num_nodes):
        """
        Generate the adjacency matrix for the graph.

        Parameters:
        edges (list): A list of edges in the graph.
        num_nodes (int): The number of nodes in the graph.

        Returns:
        ndarray: The binary adjacency matrix representing the graph.
        """
        # Initialize an adjacency matrix with zeros
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for edge in edges:
            # Fill in the adjacency matrix: edge present = 1, otherwise = 0
            A[edge] = 1.
        return A
    

class PretrainGraph:
    """
    A class to represent the graph structure of left and right hands with a 
    total of 42 keypoints.

    This class creates a graph representation of both hands, with nodes 
    representing keypoints and edges representing the connections between these
    keypoints. 

    Note that right hand goes first.

    Attributes:
    num_nodes (int): The number of nodes (keypoints) in the graph.
    edges (list): A list of edges representing connections between keypoints.
    self_loops (list): A list of self-loops for each node in the graph.
    A_binary (ndarray): The binary adjacency matrix of the graph without self-loops.
    A_binary_with_I (ndarray): The binary adjacency matrix of the graph with self-loops.
    """
    def __init__(self, *args, **kwargs):
        num_node = 42 # Total number of keypoints for both hands
        self.num_nodes = num_node
        rhand_inward = [
            (0, 1), (0, 5), (0, 17), # Wrist
            (5, 9), (9, 13), (13, 17), # Palm
            (1, 2), (2, 3), (3, 4), # Thumb
            (5, 6), (6, 7), (7, 8), # Index
            (9, 10), (10, 11), (11, 12), # Middle
            (13, 14), (14, 15), (15, 16), # Ring
            (17, 18), (18, 19), (19, 20) # Pinky
        ]
        # Define edges for the left hand, offset by 21
        lhand_inward = np.array(rhand_inward) + 21
        lhand_inward = list(map(tuple, lhand_inward))
        # Outward edges are just inward edges reversed
        rhand_outward = [(j, i) for (i, j) in rhand_inward]
        lhand_outward = [(j, i) for (i, j) in lhand_inward]
        # Combine all edges
        neighbor = rhand_inward + rhand_outward + lhand_inward + lhand_outward
        # Save the edges and self-loops
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        # Generate binary adjacency matrices
        self.A_binary = self.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = self.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)

    def get_adjacency_matrix(self, edges, num_nodes):
        """
        Generate the adjacency matrix for the graph.

        Parameters:
        edges (list): A list of edges in the graph.
        num_nodes (int): The number of nodes in the graph.

        Returns:
        ndarray: The binary adjacency matrix representing the graph.
        """
        # Initialize an adjacency matrix with zeros
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for edge in edges:
            # Fill in the adjacency matrix: edge present = 1, otherwise = 0
            A[edge] = 1.
        return A


class PretrainGestureExtractor(nn.Module):
    """
    A PyTorch module for extracting features from hand gesture data using MSG3D and STGCN.

    This module processes keypoints representing hand gestures, initially using the MSG3D model
    and optionally applying clustering and pooling for further feature extraction.

    Works like the GestureExtractor class below but handles the two-handed case
    scenario that is present during pretraining.

    Attributes:
    model (MSG3D): The Multi-Scale Spatial Temporal Graph Convolutional Network.
    maxpool1 (MediapipeHandPooling): Hand pooling layer for initial feature extraction.
    stgcn (STGCN): Spatial Temporal Graph Convolutional Network for clustered data.
    maxpool2 (MediapipeHandPooling): Additional hand pooling layer for further feature extraction.
    dropout (nn.Dropout): Dropout layer for regularization.
    """
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
            input_both_hands=False
        ):
        super().__init__()
        self.do_cluster = do_cluster
        self.relu_between = relu_between
        self.input_both_hands = input_both_hands
        # Initialize the MSG3D model
        self.model = MSG3D(
            num_point,
            num_gcn_scales,
            num_g3d_scales,
            PretrainGraph(), # Pretrain graph
            hid_dim,
            msg_3d_dropout,
            in_channels,
        )
        # Initialize clustering and pooling components if enabled
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
        # Initialize dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass for processing the input keypoints data.

        Parameters:
        x (Tensor): The input tensor containing keypoints data.

        Returns:
        tuple: Processed right and left hand features.
        """
        # Compute sequence lengths (excluding zero-padding)
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # MSG3D expects data in (N, C, T, V, M) format
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        x = self.model(x, lens)
        # Apply clustering and pooling if enabled
        if self.do_cluster:
            rhand = x[...,:21]
            lhand = x[...,21:]
            # Apply first max-pooling
            rhand = self.maxpool1(rhand)
            lhand = self.maxpool1(lhand)
            # Apply non-linearity in between if enabled
            if self.relu_between:
                rhand = F.relu(rhand) # Add M dimension
                lhand = F.relu(lhand) # Add M dimension
            rhand = rhand.unsqueeze(-1)
            lhand = lhand.unsqueeze(-1)
            # Extract features with STGCN
            rhand = self.stgcn(rhand, lens)
            lhand = self.stgcn(lhand, lens)
            rhand = rhand.squeeze(1)
            lhand = lhand.squeeze(1)
            # Apply second max-pooling
            rhand = self.maxpool2(rhand)
            lhand = self.maxpool2(lhand)
            # Apply non-linearity in between if enabled
            if self.relu_between:
                rhand = F.relu(rhand)
                lhand = F.relu(lhand)
            rhand = rhand.unsqueeze(-1) # Add M dimension
            lhand = lhand.unsqueeze(-1) # Add M dimension
            x = torch.concat((rhand, lhand), dim=3)
        else:
            x = x.unsqueeze(-1) # Add M dimension
        # Apply dropout
        x = self.dropout(x)
        rhand = x[:, :, :, 0]
        lhand = x[:, :, :, 1]

        return (rhand, lhand)


class GestureExtractor(nn.Module):
    """
    A PyTorch module for extracting features from hand gesture data using MSG3D and STGCN.

    This module processes keypoints representing hand gestures, initially using the MSG3D model
    and optionally applying clustering and pooling for further feature extraction.

    Attributes:
    model (MSG3D): The Multi-Scale Spatial Temporal Graph Convolutional Network.
    maxpool1 (MediapipeHandPooling): Hand pooling layer for initial feature extraction.
    stgcn (STGCN): Spatial Temporal Graph Convolutional Network for clustered data.
    maxpool2 (MediapipeHandPooling): Additional hand pooling layer for further feature extraction.
    dropout (nn.Dropout): Dropout layer for regularization.
    """
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
            input_both_hands=False
        ):
        super().__init__()
        self.do_cluster = do_cluster
        self.relu_between = relu_between
        self.input_both_hands = input_both_hands
        # Initialize the MSG3D model
        self.model = MSG3D(
            num_point,
            num_gcn_scales,
            num_g3d_scales,
            Hands17Graph(), # Feasibility study graph
            hid_dim,
            msg_3d_dropout,
            in_channels,
        )
        # Initialize clustering and pooling components if enabled 
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
        # Initialize dropout for regularization 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass for processing the input keypoints data.

        Parameters:
        x (Tensor): The input tensor containing keypoints data.

        Returns:
        Tensor: Processed features after passing through the network.
        """
        # Compute sequence lengths (excluding zero-padding)
        lens = (x!=0.0).all(-1).all(-1).sum(1)
        # MSG3D expects data in (N, C, T, V, M) format
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        x = self.model(x, lens)
        # Apply clustering and pooling if enabled 
        if self.do_cluster:
            # Apply first max-pooling
            x = self.maxpool1(x)
            # Apply non-linearity if enabled
            if self.relu_between:
                x = F.relu(x)
            x = x.unsqueeze(-1)
            # Extract features with STGCN
            x = self.stgcn(x, lens)
            x = x.squeeze(-1)
            # Apply second max-pooling
            x = self.maxpool2(x)
            # Apply non-linearity if enabled
            if self.relu_between:
                x = F.relu(x)            
            x = x.unsqueeze(-1) # Add M dimension
        else:
            x = x.unsqueeze(-1) # Add M dimension
        # Apply dropout
        x = self.dropout(x)

        return x