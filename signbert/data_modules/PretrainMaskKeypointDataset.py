from multiprocessing import Lock

import torch
import numpy as np
from torch.utils.data import Dataset

from signbert.data_modules.utils import mask_transform, mask_transform_identity

from IPython import embed; from sys import exit

file_lock = Lock()


class PretrainMaskKeypointDataset(Dataset):

    def __init__(
            self, 
            idxs_fpath, 
            npy_fpath, 
            R, 
            m, 
            K, 
            max_disturbance=0.25, 
            identity=False,
            no_mask_joint=False,
            openpose=False
        ):
        """In the paper they perform an ablation on the MSASL dataset:
            - R: 40%
            - m: not provided
            - K: 8
        """
        super().__init__()
        with file_lock:
            self.idxs = np.load(idxs_fpath)
            self.data = np.load(npy_fpath)
        # Max. number of frames to mask, ablation study, 0.4
        self.R = R
        # Number of joints to take when performing joint masking, ablation study
        self.m = m
        # Max. number of continuous frames to take when performing clip masking, ablation study 8
        self.K = K
        # Max disturbance to add in joint masking
        self.max_disturbance = max_disturbance
        self.identity = identity
        self.no_mask_joint = no_mask_joint
        self.openpose = openpose

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq_idx = self.idxs[idx]
        seq = self.data[idx]
        score = seq[...,-1]
        seq = seq[...,:-1]
        # MSASL dataset has Openpose keypoints computed with scores
        if self.openpose:
            arms = seq[:, (82, 79, 83, 80, 84, 81)]
            lhand = seq[:, 95:116]
            rhand = seq[:, 116:]
            lhand_scores = score[:, 95:116]
            rhand_scores = score[:, 116:]
        else:
            arms = seq[:, 5:11]
            lhand = seq[:, 91:112]
            rhand = seq[:, 112:133]
            lhand_scores = score[:, 91:112]
            rhand_scores = score[:, 112:133]
        if self.identity:
            rhand_masked, rhand_masked_frames_idx = mask_transform_identity(rhand, self.R, self.max_disturbance, self.no_mask_joint, self.K, self.m)
            lhand_masked, lhand_masked_frames_idx = mask_transform_identity(lhand, self.R, self.max_disturbance, self.no_mask_joint, self.K, self.m)
        else:
            rhand_masked, rhand_masked_frames_idx = mask_transform(rhand, self.R, self.max_disturbance, self.no_mask_joint, self.K, self.m)
            lhand_masked, lhand_masked_frames_idx = mask_transform(lhand, self.R, self.max_disturbance, self.no_mask_joint, self.K, self.m)

        return (
            seq_idx, 
            arms,
            rhand, 
            rhand_masked,
            rhand_masked_frames_idx,
            rhand_scores,
            lhand, 
            lhand_masked,
            lhand_masked_frames_idx,
            lhand_scores,
        )


def mask_keypoint_dataset_collate_fn(batch):
    """
    Custom DataLoader collate function.

    Adds padding and changes data format so it can be batched.
    """
    seq_idxs = [] 
    arms_seqs = []
    rhand_seqs = []
    rhand_masked_seqs = []
    rhand_masked_frames_idx_seqs = []
    rhand_scores_seqs = []
    lhand_seqs = [] 
    lhand_masked_seqs = []
    lhand_masked_frames_idx_seqs = []
    lhand_scores_seqs = []
    # Find masked frames indices pad values 
    rhand_n_masked_frames_idxs = np.array([len(b[4]) for b in batch])
    rhand_pad_value = rhand_n_masked_frames_idxs.max() - rhand_n_masked_frames_idxs
    lhand_n_masked_frames_idxs = np.array([len(b[8]) for b in batch])
    lhand_pad_value = lhand_n_masked_frames_idxs.max() - lhand_n_masked_frames_idxs
    for i in range(len(batch)):
        (seq_idx, 
        arms,
        rhand, 
        rhand_masked,
        rhand_masked_frames_idx,
        rhand_scores,
        lhand, 
        lhand_masked,
        lhand_masked_frames_idx,
        lhand_scores) = batch[i]

        seq_idxs.append(seq_idx) 
        arms_seqs.append(arms)
        rhand_seqs.append(rhand)
        rhand_masked_seqs.append(rhand_masked)
        rhand_masked_frames_idx_seqs.append(np.pad(rhand_masked_frames_idx, (0, rhand_pad_value[i]), mode='constant', constant_values=-1.))
        rhand_scores_seqs.append(rhand_scores)
        lhand_seqs.append(lhand) 
        lhand_masked_seqs.append(lhand_masked)
        lhand_masked_frames_idx_seqs.append(np.pad(lhand_masked_frames_idx, (0, lhand_pad_value[i]), mode='constant', constant_values=-1.))
        lhand_scores_seqs.append(lhand_scores)
        
    seq_idxs = np.array(seq_idxs) 
    arms_seqs = np.stack(arms_seqs)
    rhand_seqs = np.stack(rhand_seqs)
    rhand_masked_seqs = np.stack(rhand_masked_seqs)
    rhand_masked_frames_idx_seqs = np.stack(rhand_masked_frames_idx_seqs)
    rhand_scores_seqs = np.stack(rhand_scores_seqs)
    lhand_seqs = np.stack(lhand_seqs) 
    lhand_masked_seqs = np.stack(lhand_masked_seqs)
    lhand_masked_frames_idx_seqs = np.stack(lhand_masked_frames_idx_seqs)
    lhand_scores_seqs = np.stack(lhand_scores_seqs)
    
    seq_idxs = torch.tensor(seq_idxs, dtype=torch.int32) 
    arms_seqs = torch.tensor(arms_seqs, dtype=torch.float32)
    rhand_seqs = torch.tensor(rhand_seqs, dtype=torch.float32)
    rhand_masked_seqs = torch.tensor(rhand_masked_seqs, dtype=torch.float32)
    rhand_masked_frames_idx_seqs = torch.tensor(rhand_masked_frames_idx_seqs, dtype=torch.int64)
    rhand_scores_seqs = torch.tensor(rhand_scores_seqs, dtype=torch.float32)
    lhand_seqs = torch.tensor(lhand_seqs, dtype=torch.float32) 
    lhand_masked_seqs = torch.tensor(lhand_masked_seqs, dtype=torch.float32)
    lhand_masked_frames_idx_seqs = torch.tensor(lhand_masked_frames_idx_seqs, dtype=torch.int64)
    lhand_scores_seqs = torch.tensor(lhand_scores_seqs, dtype=torch.float32)

    return (
        seq_idxs,
        arms_seqs,
        rhand_seqs,
        rhand_masked_seqs,
        rhand_masked_frames_idx_seqs,
        rhand_scores_seqs,
        lhand_seqs,
        lhand_masked_seqs,
        lhand_masked_frames_idx_seqs,
        lhand_scores_seqs,
    )