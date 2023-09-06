import torch
import numpy as np

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from IPython import embed; from sys import exit

class MaskKeypointDataset(Dataset):

    def __init__(self, npy_fpath, R, m, K):
        """In the paper they perform an ablation on the MSASL dataset:
            - R: 40%
            - m: not provided
            - K: 8
        """
        super().__init__()

        self.data = np.load(npy_fpath)
        # Max. number of frames to mask, ablation study, 0.4
        self.R = R
        # Number of joints to take when performing joint masking, ablation study
        self.m = m
        # Max. number of continuous frames to take when performing clip masking, ablation study 8
        self.K = K
        # Max disturbance to add in joint masking (in pixels)
        self.max_disturbance = 5

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        seq, masked_frames_idx = self.mask_transform(seq)

        return seq, masked_frames_idx

    def mask_transform(self, seq):
        # Get the number of frames without masking
        n_frames = (seq != 0.0).all((1,2)).sum()
        # Get the total number of frames to mask
        n_frames_to_mask = int(np.ceil(self.R * n_frames))
        # Get the actual frame idxs to mask
        frames_to_mask = np.random.choice(n_frames, size=n_frames_to_mask, replace=False)

        clipped_masked_frames = []
        for f in frames_to_mask:
            # Grab frame to be masked
            curr_frame = seq[f]
            # Select the type of masking
            # 0 joint masking; 1 frame masking; 2 clip masking; 3 identity
            # This could be a point of deviation. Do the authors consider the 
            # identity to be taken into account into the loss?
            op_idx = np.random.choice(4)
            
            if op_idx == 0:
                # Joint masking
                curr_frame = self.mask_joint(curr_frame)
            elif op_idx == 1:
                # Frame masking
                curr_frame[:] = 0.
            elif op_idx == 2:
                # Clip masking
                curr_frame, masked_frames_idx = self.mask_clip(f, seq, n_frames)
                # Add masked clip frames
                clipped_masked_frames.extend(masked_frames_idx)
            else:
                # Identity
                pass
        # Used in loss calculation, only masked frames are taken into account
        masked_frames_idx = np.unique(np.concatenate((frames_to_mask, clipped_masked_frames)))
        
        return seq, masked_frames_idx

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
        # Select the joints to mask
        joint_idxs_to_mask = np.random.choice(21, size=self.m, replace=False)
        # Select the operation
        # 0 zero-masking; 1 spatial disturbance
        op_idx = np.random.binomial(1, p=0.5, size=self.m).reshape(-1, 1)

        def spatial_disturbance(xy):
            return xy + [np.random.uniform(-self.max_disturbance, self.max_disturbance), np.random.uniform(-self.max_disturbance, self.max_disturbance)]

        # move this to another place, useful for testing
        # crop where the hand is
        # print(f'{joint_idxs_to_mask=}')
        # print(f'{op_idx=}')
        # fig, axx = plt.subplots(1, 2, dpi=300)
        # axx[0].imshow(np.zeros((480, 640)), cmap='gray')
        # axx[0].scatter(frame[:,0], frame[:,1], c='r', s=2)
        # for i in range(len(frame)):
        #     axx[0].text(frame[i,0], frame[i,1], str(i), ha='center', va='bottom', fontsize=3, color='white')
        # axx[0].axis('off')
        # axx[0].set_title('Before')

        frame[joint_idxs_to_mask] = np.where(op_idx, spatial_disturbance(frame[joint_idxs_to_mask]), 0.0)

        # axx[1].imshow(np.zeros((480, 640)), cmap='gray')
        # axx[1].scatter(frame[:,0], frame[:,1], c='r', s=2)
        # for i in range(len(frame)):
        #     axx[1].text(frame[i,0], frame[i,1], str(i), ha='center', va='bottom', fontsize=3, color='white')
        # axx[1].axis('off')
        # axx[1].set_title('After')
        # plt.tight_layout()
        # fig.savefig('frame-masked.png')
        # plt.close()

        return frame
    
def mask_keypoint_dataset_collate_fn(batch):
    seqs = []
    masked_frame_idxs = []
    # Get number of masked frames idxs
    n_masked_frames_idxs = [len(b[1]) for b in batch]
    # Find max number of masked frames idxs
    max_masked_frames = max(n_masked_frames_idxs)
    # Find padding value
    pad_value = max_masked_frames - np.array(n_masked_frames_idxs)
    # Pad masked frames idxs
    for i in range(len(batch)):
        seq, frame_idxs = batch[i]
        seqs.append(seq)
        masked_frame_idxs.append(np.pad(frame_idxs, (0, pad_value[i]), mode='constant', constant_values=-1.))
    seqs = np.stack(seqs)
    masked_frame_idxs = np.stack(masked_frame_idxs)
    seqs = torch.tensor(seqs)        
    masked_frame_idxs = torch.tensor(masked_frame_idxs)
    
    return (seqs, masked_frame_idxs)

if __name__ == '__main__':
    import os
    import time

    from torch.utils.data import DataLoader

    dataset = MaskKeypointDataset(
        npy_fpath='/home/gts/projects/jsoutelo/SignBERT+/datasets/HANDS17/preprocess/X_train.npy',
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
    