import torch
import numpy as np

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from multiprocessing import Lock
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
            no_mask_joint=False
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

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq_idx = self.idxs[idx]
        seq = self.data[idx]
        score = seq[...,-1]
        seq = seq[...,:-1]
        arms = seq[:, 5:11]
        lhand = seq[:, 91:112]
        rhand = seq[:, 112:133]
        lhand_scores = score[:, 91:112]
        rhand_scores = score[:, 112:133]
        if self.identity:
            rhand_masked, rhand_masked_frames_idx = self.mask_transform_identity(rhand)
            lhand_masked, lhand_masked_frames_idx = self.mask_transform_identity(lhand)
        else:
            rhand_masked, rhand_masked_frames_idx = self.mask_transform(rhand)
            lhand_masked, lhand_masked_frames_idx = self.mask_transform(lhand)

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

    def mask_transform_identity(self, seq):
        toret = seq.copy()
        # Get the number of frames without masking
        n_frames = (toret != 0.0).all((1,2)).sum()
        # Get the total number of frames to mask
        n_frames_to_mask = int(np.ceil(self.R * n_frames))
        # Get the actual frame idxs to mask
        frames_to_mask = np.random.choice(n_frames, size=n_frames_to_mask, replace=False)

        clipped_masked_frames = []
        for f in frames_to_mask:
            # Grab frame to be masked
            curr_frame = toret[f]
            # Select the type of masking
            # 0 joint masking; 1 frame masking; 2 clip masking; 3 identity
            # This could be a point of deviation. Do the authors consider the 
            # identity to be taken into account into the loss?
            op_idx = np.random.choice(4)
            
            if op_idx == 0:
                # Joint masking
                curr_frame = self.mask_joint(curr_frame)
                toret[f] = curr_frame
            elif op_idx == 1:
                # Frame masking
                curr_frame[:] = 0.
                toret[f] = curr_frame
            elif op_idx == 2:
                # Clip masking
                curr_frame, masked_frames_idx = self.mask_clip(f, toret, n_frames)
                # Add masked clip frames
                clipped_masked_frames.extend(masked_frames_idx)
            else:
                # Identity; do nothing
                pass
        # Used in loss calculation, only masked frames are taken into account
        masked_frames_idx = np.unique(np.concatenate((frames_to_mask, clipped_masked_frames)))
        
        return toret, masked_frames_idx

    def mask_transform(self, seq):
        toret = seq.copy()
        # Get the number of frames without masking
        n_frames = (toret != 0.0).all((1,2)).sum()
        # Get the total number of frames to mask
        n_frames_to_mask = int(np.ceil(self.R * n_frames))
        # Get the actual frame idxs to mask
        frames_to_mask = np.random.choice(n_frames, size=n_frames_to_mask, replace=False)

        clipped_masked_frames = []
        for f in frames_to_mask:
            # Grab frame to be masked
            curr_frame = toret[f]
            # Select the type of masking
            # 0 joint masking; 1 frame masking; 2 clip masking; 3 identity
            # This could be a point of deviation. Do the authors consider the 
            # identity to be taken into account into the loss?
            op_idx = np.random.choice(3)
            
            if op_idx == 0:
                # Joint masking
                curr_frame = self.mask_joint(curr_frame)
                toret[f] = curr_frame
            elif op_idx == 1:
                # Frame masking
                curr_frame[:] = 0.
                toret[f] = curr_frame
            else:
                # Clip masking
                curr_frame, masked_frames_idx = self.mask_clip(f, toret, n_frames)
                # Add masked clip frames
                clipped_masked_frames.extend(masked_frames_idx)
        # Used in loss calculation, only masked frames are taken into account
        masked_frames_idx = np.unique(np.concatenate((frames_to_mask, clipped_masked_frames)))
        
        return toret, masked_frames_idx

    def mask_clip(self, frame_idx, seq, n_frames):
        n_frames_to_mask = np.random.randint(2, self.K+1)
        n_frames_to_mask_half = n_frames_to_mask // 2

        start_idx = frame_idx - n_frames_to_mask_half
        end_idx = frame_idx + (n_frames_to_mask - n_frames_to_mask_half)

        if start_idx < 0:
            diff = abs(start_idx)
            start_idx = 0
            end_idx += diff
        if end_idx > n_frames:
            diff = end_idx - n_frames
            end_idx = n_frames
            start_idx -= diff

        masked_frames_idx = list(range(start_idx, end_idx))
        seq[masked_frames_idx] = 0.0

        return seq, masked_frames_idx

    def mask_joint(self, frame):
        m = np.random.randint(1, self.m+1)
        # Select the joints to mask
        joint_idxs_to_mask = np.random.choice(21, size=m, replace=False)
        # Select the operation
        # 0 zero-masking; 1 spatial disturbance
        op_idx = np.random.binomial(1, p=0.5, size=m).reshape(-1, 1)

        def spatial_disturbance(xy):
            return xy + [np.random.uniform(-self.max_disturbance, self.max_disturbance), np.random.uniform(-self.max_disturbance, self.max_disturbance)]

        frame[joint_idxs_to_mask] = np.where(
            op_idx, 
            spatial_disturbance(frame[joint_idxs_to_mask]), 
            spatial_disturbance(frame[joint_idxs_to_mask]) if self.no_mask_joint else 0.0
        )

        return frame
    
def mask_keypoint_dataset_collate_fn(batch):
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