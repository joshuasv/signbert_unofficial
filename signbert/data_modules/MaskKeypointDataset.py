from multiprocessing import Lock

import torch
import numpy as np
from torch.utils.data import Dataset

from signbert.data_modules.utils import mask_transform, mask_transform_identity

from IPython import embed; from sys import exit


file_lock = Lock()

class MaskKeypointDataset(Dataset):

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
        if self.identity:
            seq_masked, masked_frames_idx = mask_transform_identity(seq, self.R, self.max_disturbance, self.no_mask_joint, self.K, self.m)
        else:
            seq_masked, masked_frames_idx = mask_transform(seq, self.R, self.max_disturbance, self.no_mask_joint, self.K, self.m)

        return (seq_idx, seq, seq_masked, score, masked_frames_idx)
    

def mask_keypoint_dataset_collate_fn(batch):
    """
    Custom DataLoader collate function.

    Adds padding and changes data format so it can be batched.
    """ 
    idxs = []
    seqs = []
    seqs_masked = []
    scores = []
    masked_frame_idxs = []
    # Get number of masked frames idxs
    n_masked_frames_idxs = [len(b[4]) for b in batch]
    # Find max number of masked frames idxs
    max_masked_frames = max(n_masked_frames_idxs)
    # Find padding value
    pad_value = max_masked_frames - np.array(n_masked_frames_idxs)
    # Pad masked frames idxs
    for i in range(len(batch)):
        idx, seq, seq_masked, score, frame_idxs = batch[i]
        idxs.append(idx)
        seqs.append(seq)
        seqs_masked.append(seq_masked)
        scores.append(score)
        masked_frame_idxs.append(np.pad(frame_idxs, (0, pad_value[i]), mode='constant', constant_values=-1.))
    idxs = np.array(idxs)
    seqs = np.stack(seqs)
    seqs_masked = np.stack(seqs_masked)
    scores = np.stack(scores)
    masked_frame_idxs = np.stack(masked_frame_idxs)
    idxs = torch.tensor(idxs, dtype=torch.int32)
    seqs = torch.tensor(seqs, dtype=torch.float32)      
    seqs_masked = torch.tensor(seqs_masked, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)  
    masked_frame_idxs = torch.tensor(masked_frame_idxs, dtype=torch.int64)

    return (idxs, seqs, seqs_masked, scores, masked_frame_idxs)

if __name__ == '__main__':
    import os
    import time

    from torch.utils.data import DataLoader

    dataset = MaskKeypointDataset(
        idxs_fpath='/home/gts/projects/jsoutelo/SignBERT+/datasets/HANDS17/preprocess/idxs.npy',
        npy_fpath='/home/temporal2/jsoutelo/datasets/HANDS17/preprocess/X_train.npy',
        R=0.2,
        m=5,
        K=6
    )
    sample = dataset[0]

    # Visualize

    # Profile DataLoader
    num_samples = 10
    batch_size = 32
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        # num_workers=os.cpu_count(), # Works faster without it
        collate_fn=mask_keypoint_dataset_collate_fn
    )
    elapsed_record = []
    for _ in range(num_samples):
        start_time = time.time()
        for batch in dataloader:
            pass
        elapsed = time.time() - start_time
        elapsed_record.append(elapsed)
        print(f"Time taken for {len(dataset)} samples: {elapsed:.4f} seconds")
    # Calculate average duration
    average_duration = np.mean(elapsed_record)
    print(f"Average time per __getitem__: {average_duration:.4f} seconds") # 0.8775 seconds
    