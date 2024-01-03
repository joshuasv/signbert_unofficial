import os
import gc
import glob

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from signbert.data_modules.PretrainMaskKeypointDataset import PretrainMaskKeypointDataset, mask_keypoint_dataset_collate_fn
from signbert.utils import read_txt_as_list, dict_to_json_file

from IPython import embed


class MSASLDataModule(pl.LightningDataModule):

    DPATH = '/home/tmpvideos/SLR/MSASL'
    TRAIN_SPLIT_JSON_FPATH = os.path.join(DPATH, 'MSASL_train.json')
    VAL_SPLIT_JSON_FPATH = os.path.join(DPATH, 'MSASL_val.json')
    TEST_SPLIT_JSON_FPATH = os.path.join(DPATH, 'MSASL_test.json')
    CLASSES_JSON_FPATH = os.path.join(DPATH, 'MSASL_classes.json')
    MISSING_VIDEOS_FPATH = os.path.join(DPATH, 'raw_videos', 'missing.txt')
    SKELETON_DPATH = os.path.join(DPATH, 'skeleton-data', 'rtmpose-l_8xb64-270e_coco-wholebody-256x192')
    TRAIN_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'train')
    VAL_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'val')
    TEST_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'test')
    PREPROCESS_DPATH = os.path.join(DPATH, 'preprocess')
    MEANS_FPATH = os.path.join(PREPROCESS_DPATH, 'means.npy')
    STDS_FPATH = os.path.join(PREPROCESS_DPATH, 'stds.npy')
    TRAIN_FPATH = os.path.join(PREPROCESS_DPATH, 'train.npy')
    VAL_FPATH = os.path.join(PREPROCESS_DPATH, 'val.npy')
    TEST_FPATH = os.path.join(PREPROCESS_DPATH, 'test.npy')
    TRAIN_NORM_FPATH = os.path.join(PREPROCESS_DPATH, 'train_norm.npy')
    VAL_NORM_FPATH = os.path.join(PREPROCESS_DPATH, 'val_norm.npy')
    TEST_NORM_FPATH = os.path.join(PREPROCESS_DPATH, 'test_norm.npy')
    TRAIN_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'train_idxs.npy')
    VAL_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'val_idxs.npy')
    TEST_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'test_idxs.npy')
    TRAIN_MAPPING_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'train_mapping_idxs.json')
    VAL_MAPPING_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'val_mapping_idxs.json')
    TEST_MAPPING_IDXS_FPATH = os.path.join(PREPROCESS_DPATH, 'test_mapping_idxs.json')
    SEQ_PAD_VALUE = 0.0

    def __init__(self, batch_size, normalize=False, R=0.3, m=5, K=8, max_disturbance=0.25):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.R = R
        self.m = m
        self.K = K
        self.max_disturbance = max_disturbance
        self.means_fpath = MSASLDataModule.MEANS_FPATH
        self.stds_fpath = MSASLDataModule.STDS_FPATH

    def prepare_data(self):
        # Create preprocess path if it does not exist
        if not os.path.exists(MSASLDataModule.PREPROCESS_DPATH):
            os.makedirs(MSASLDataModule.PREPROCESS_DPATH)

        # Compute means and stds
        if not os.path.exists(MSASLDataModule.MEANS_FPATH) or \
            not os.path.exists(MSASLDataModule.STDS_FPATH):
            # Grab skeleton file paths
            skeleton_fpaths = glob.glob(
                os.path.join(MSASLDataModule.TRAIN_SKELETON_DPATH, '*.npy')
            )
            # Load data filtering missing
            train_idxs = [os.path.basename(f).split('.npy')[0] for f in skeleton_fpaths]
            missing_idxs = read_txt_as_list(MSASLDataModule.MISSING_VIDEOS_FPATH)
            train = [
                np.load(f)
                for idx, f in zip(train_idxs, skeleton_fpaths)
                if idx not in missing_idxs
            ]
            self._generate_means_stds(train)
        
        if not os.path.exists(MSASLDataModule.TRAIN_FPATH) or \
            not os.path.exists(MSASLDataModule.VAL_FPATH) or \
            not os.path.exists(MSASLDataModule.TEST_FPATH) or \
            not os.path.exists(MSASLDataModule.TRAIN_NORM_FPATH) or \
            not os.path.exists(MSASLDataModule.VAL_NORM_FPATH) or \
            not os.path.exists(MSASLDataModule.TEST_NORM_FPATH) or \
            not os.path.exists(MSASLDataModule.TRAIN_IDXS_FPATH) or \
            not os.path.exists(MSASLDataModule.VAL_IDXS_FPATH) or \
            not os.path.exists(MSASLDataModule.TEST_IDXS_FPATH) or \
            not os.path.exists(MSASLDataModule.TRAIN_MAPPING_IDXS_FPATH) or \
            not os.path.exists(MSASLDataModule.VAL_MAPPING_IDXS_FPATH) or \
            not os.path.exists(MSASLDataModule.TEST_MAPPING_IDXS_FPATH):
            # Grab missing idxs
            missing_idxs = read_txt_as_list(MSASLDataModule.MISSING_VIDEOS_FPATH)
            # Grab data file paths and filter missing
            train_skeleton_fpaths = glob.glob(
                os.path.join(MSASLDataModule.TRAIN_SKELETON_DPATH, '*.npy')
            )
            train_idxs = [os.path.basename(f).split('.npy')[0] for f in train_skeleton_fpaths]
            train_skeleton_fpaths = [
                f 
                for idx, f in zip(train_idxs, train_skeleton_fpaths) 
                if idx not in missing_idxs
            ]  
            train_idxs = [idx for idx in train_idxs if idx not in missing_idxs]
            val_skeleton_fpaths = glob.glob(
                os.path.join(MSASLDataModule.VAL_SKELETON_DPATH, '*.npy')
            )
            val_idxs = [os.path.basename(f).split('.npy')[0] for f in val_skeleton_fpaths]
            val_skeleton_fpaths = [
                f 
                for idx, f in zip(val_idxs, val_skeleton_fpaths) 
                if idx not in missing_idxs
            ]  
            val_idxs = [idx for idx in train_idxs if idx not in missing_idxs]
            test_skeleton_fpaths = glob.glob(
                os.path.join(MSASLDataModule.TEST_SKELETON_DPATH, '*.npy')
            )
            test_idxs = [os.path.basename(f).split('.npy')[0] for f in test_skeleton_fpaths]
            test_skeleton_fpaths = [
                f 
                for idx, f in zip(test_idxs, test_skeleton_fpaths) 
                if idx not in missing_idxs
            ]  
            test_idxs = [idx for idx in test_idxs if idx not in missing_idxs]
            # Generate Numpy
            self._generate_preprocess_npy_arrays(
                train_idxs, 
                train_skeleton_fpaths, 
                MSASLDataModule.TRAIN_FPATH, 
                MSASLDataModule.TRAIN_NORM_FPATH,
                MSASLDataModule.TRAIN_IDXS_FPATH,
                MSASLDataModule.TRAIN_MAPPING_IDXS_FPATH
            )
            self._generate_preprocess_npy_arrays(
                val_idxs, 
                val_skeleton_fpaths, 
                MSASLDataModule.VAL_FPATH, 
                MSASLDataModule.VAL_NORM_FPATH,
                MSASLDataModule.VAL_IDXS_FPATH,
                MSASLDataModule.VAL_MAPPING_IDXS_FPATH
            )
            self._generate_preprocess_npy_arrays(
                test_idxs, 
                test_skeleton_fpaths, 
                MSASLDataModule.TEST_FPATH, 
                MSASLDataModule.TEST_NORM_FPATH,
                MSASLDataModule.TEST_IDXS_FPATH,
                MSASLDataModule.TEST_MAPPING_IDXS_FPATH
            )
            
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            X_train_fpath = MSASLDataModule.TRAIN_NORM_FPATH if self.normalize else MSASLDataModule.TRAIN_FPATH
            X_val_fpath = MSASLDataModule.VAL_NORM_FPATH if self.normalize else MSASLDataModule.VAL_FPATH
            X_test_fpath = MSASLDataModule.TEST_NORM_FPATH if self.normalize else MSASLDataModule.TEST_FPATH

            self.setup_train = PretrainMaskKeypointDataset(
                MSASLDataModule.TRAIN_IDXS_FPATH, 
                X_train_fpath, 
                self.R, 
                self.m, 
                self.K, 
                self.max_disturbance
            )
            self.setup_val = PretrainMaskKeypointDataset(
                MSASLDataModule.VAL_IDXS_FPATH,
                X_val_fpath, 
                self.R, 
                self.m, 
                self.K, 
                self.max_disturbance
            )

    def train_dataloader(self):
        return DataLoader(self.setup_train, batch_size=self.batch_size, collate_fn=mask_keypoint_dataset_collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.setup_val, batch_size=self.batch_size, collate_fn=mask_keypoint_dataset_collate_fn)

    def _generate_means_stds(self, train_data):
        """Compute mean and standard deviation for all x and y coordinates."""
        seq_concats = np.concatenate([s[..., :2] for s in train_data], axis=0)
        means = seq_concats.mean((0, 1))
        stds = seq_concats.std((0, 1))
        np.save(MSASLDataModule.MEANS_FPATH, means)
        np.save(MSASLDataModule.STDS_FPATH, stds)

    def _generate_preprocess_npy_arrays(
        self, 
        split_idxs, 
        skeleton_fpaths, 
        out_fpath,
        norm_out_fpath,
        idxs_out_fpath,
        idxs_mapping_out_fpath,
        max_seq_len=500
    ):
        """
        Process and save sequences of skeleton data.

        This function handles sequence splitting if they exceed a maximum length, normalization, 
        padding to a uniform length, and maintains a mapping of processed sequences to their
        original indices. It then saves these processed sequences in .npy format for efficient access.

        Parameters:
        split_idxs (list): Indices indicating where to split the sequences.
        skeleton_fpaths (list): File paths of raw skeleton sequences.
        out_fpath (str): File path for saving processed sequences.
        norm_out_fpath (str): File path for saving normalized sequences.
        idxs_out_fpath (str): File path for saving the indices of sequences.
        idxs_mapping_out_fpath (str): File path for saving the mapping indices.
        max_seq_len (int): Maximum length of a sequence before splitting. Default is 500.
        """
        seqs = []
        sequential_idx = []  # To track the order of sequences
        counter = 0  # For assigning new indices to split sequences
        mapping_idxs = {}  # To map new indices to original sequence indices
        # Iterate over each file path and its corresponding index
        for idx, f in zip(split_idxs, skeleton_fpaths):
            seq = np.load(f)
            # Check if the sequence exceeds the max length and needs splitting
            if seq.shape[0] > max_seq_len:
                split_indices = list(range(max_seq_len, seq.shape[0], max_seq_len))
                seq = np.array_split(seq, split_indices, axis=0)
                for s in seq:
                    seqs.append(s)
                    sequential_idx.append(counter)
                    mapping_idxs[counter] = idx  # Map new index to original index
                    counter += 1
            else:
                seqs.append(seq)
                sequential_idx.append(counter)
                mapping_idxs[counter] = idx
                counter += 1
        # Normalize the sequences
        seqs_norm = self._normalize_seqs(seqs)
        # Pad sequences to uniform length
        seqs = self._pad_seqs_by_max_len(seqs)
        seqs_norm = self._pad_seqs_by_max_len(seqs_norm)
        # Convert sequences to float32 format
        seqs = seqs.astype(np.float32)
        seqs_norm = seqs_norm.astype(np.float32)
        # Convert the sequence indices to int32
        seqs_idxs = np.array(sequential_idx, dtype=np.int32)
        # Save the processed sequences and their indices
        np.save(out_fpath, seqs)
        np.save(norm_out_fpath, seqs_norm)
        np.save(idxs_out_fpath, seqs_idxs)
        dict_to_json_file(mapping_idxs, idxs_mapping_out_fpath)  # Save the mapping as a JSON file
        # Clean up to free memory
        del seqs
        del seqs_norm
        del seqs_idxs
        gc.collect()

    
    def _normalize_seqs(self, seqs):
        """
        Normalize the sequences using pre-calculated means and standard deviations.

        This function normalizes each sequence in the provided list of sequences (seqs)
        by subtracting the mean and dividing by the standard deviation for each element
        in the sequence. This is a common preprocessing step in many machine learning
        tasks, as it standardizes data to have a mean of 0 and a standard deviation of 1.

        Parameters:
        seqs (list): List of sequences to be normalized.
        
        Returns:
        list: A list of normalized sequences.
        """
        # Load the pre-calculated means and standard deviations for normalization
        means = np.load(MSASLDataModule.MEANS_FPATH)
        stds = np.load(MSASLDataModule.STDS_FPATH)
        # Append a zero to means and a one to stds for the identity operation.
        # This ensures that the last element of each sequence is not affected by normalization.
        means = np.concatenate((means, [0]), -1)
        stds = np.concatenate((stds, [1]), -1)
        # Normalize each sequence in the list
        # The normalization is done by subtracting the mean and dividing by the standard deviation
        # for each element in the sequence.
        seqs_norm = [(s - means) / stds for s in seqs]

        return seqs_norm

    def _pad_seqs_by_max_len(self, seqs):
        """
        Pad all sequences in the list to the same maximum length.

        This function pads each sequence in the list 'seqs' to ensure they all have the
        same length. This is particularly useful for batch processing in machine learning
        models, as it requires all inputs to be of the same size.

        Parameters:
        seqs (list): List of sequences (numpy arrays) to be padded.

        Returns:
        numpy.ndarray: A numpy array of sequences, all padded to the same maximum length.
        """
        # Calculate the length of each sequence in the list
        seqs_len = [len(t) for t in seqs]
        # Find the maximum sequence length
        max_seq_len = max(seqs_len)
        # Define a lambda function for generating padding configuration
        # It calculates how much padding is needed for each sequence to match the max length
        lmdb_gen_pad_seq = lambda s_len: ((0,max_seq_len-s_len), (0,0), (0,0))
        # Pad each sequence in the list
        # The padding is applied only to the sequence length dimension and
        # filled with a constant value 
        seqs = np.stack([
            np.pad(
                array=t, 
                pad_width=lmdb_gen_pad_seq(seqs_len[i]),
                mode='constant',
                constant_values=MSASLDataModule.SEQ_PAD_VALUE
            ) 
            for i, t in enumerate(seqs)
        ])

        return seqs


if __name__ == '__main__':

    d = MSASLDataModule(
        batch_size=32,
        normalize=True,
    )
    d.prepare_data()
    d.setup()
    dl = d.train_dataloader()
    sample = next(iter(dl))
    embed()