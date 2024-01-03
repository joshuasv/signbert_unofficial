import os
import gc
import glob

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from signbert.data_modules.PretrainMaskKeypointDataset import PretrainMaskKeypointDataset, mask_keypoint_dataset_collate_fn

from IPython import embed


class RwthPhoenixDataModule(pl.LightningDataModule):
    DPATH = '/home/tmpvideos/SLR/RWTH-PHOENIX-Weather/phoenix2014-release/phoenix-2014-multisigner/features/skeleton-fullFrame-210x260px/rtmpose-l_8xb64-270e_coco-wholebody-256x192'
    DPATH_T = '/home/tmpvideos/SLR/PHOENIX-2014-T/features/skeleton-fullFrame-210x260px/rtmpose-l_8xb64-270e_coco-wholebody-256x192'
    TRAIN_DPATH = os.path.join(DPATH, 'train')
    TEST_DPATH = os.path.join(DPATH, 'test')
    DEV_DPATH = os.path.join(DPATH, 'dev')
    TRAIN_DPATH_T = os.path.join(DPATH_T, 'train')
    TEST_DPATH_T = os.path.join(DPATH_T, 'test')
    DEV_DPATH_T = os.path.join(DPATH_T, 'dev')
    SEQ_PAD_VALUE = 0.0

    def __init__(self, batch_size, normalize=False, R=0.3, m=5, K=8, max_disturbance=0.25, phoenix_T=False):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.R = R
        self.m = m
        self.K = K
        self.max_disturbance = max_disturbance
        self.phoenix_T = phoenix_T
        self.dpath = RwthPhoenixDataModule.DPATH_T if phoenix_T else RwthPhoenixDataModule.DPATH
        self.train_dpath = RwthPhoenixDataModule.TRAIN_DPATH_T if phoenix_T else RwthPhoenixDataModule.TRAIN_DPATH
        self.test_dpath = RwthPhoenixDataModule.TEST_DPATH_T if phoenix_T else RwthPhoenixDataModule.TEST_DPATH
        self.dev_dpath = RwthPhoenixDataModule.DEV_DPATH_T if phoenix_T else RwthPhoenixDataModule.DEV_DPATH
        self.preprocess_dpath = os.path.join(self.dpath, 'preprocess')
        self.train_fpath = os.path.join(self.preprocess_dpath, 'train.npy')
        self.test_fpath = os.path.join(self.preprocess_dpath, 'test.npy')
        self.val_fpath = os.path.join(self.preprocess_dpath, 'val.npy')
        self.train_norm_fpath = os.path.join(self.preprocess_dpath, 'train_norm.npy')
        self.test_norm_fpath = os.path.join(self.preprocess_dpath, 'test_norm.npy')
        self.val_norm_fpath = os.path.join(self.preprocess_dpath, 'val_norm.npy')
        self.train_idxs_fpath = os.path.join(self.preprocess_dpath, 'train_idxs.npy')
        self.test_idxs_fpath = os.path.join(self.preprocess_dpath, 'test_idxs.npy')
        self.val_idxs_fpath = os.path.join(self.preprocess_dpath, 'val_idxs.npy')
        self.means_fpath = os.path.join(self.preprocess_dpath, 'means.npy')
        self.stds_fpath = os.path.join(self.preprocess_dpath, 'stds.npy')

    def prepare_data(self):
        # Create preprocess path if it does not exist
        if not os.path.exists(self.preprocess_dpath):
            os.makedirs(self.preprocess_dpath)
        
        # Compute x and y means of training keypoints
        if not os.path.exists(self.means_fpath) or \
            not os.path.exists(self.stds_fpath):
            self._generate_train_means_stds()
            
        # Check if train, validation, and test Numpy arrays exist
        if not os.path.exists(self.train_fpath) or \
            not os.path.exists(self.train_norm_fpath) or \
            not os.path.exists(self.val_fpath) or \
            not os.path.exists(self.val_norm_fpath) or \
            not os.path.exists(self.test_fpath) or \
            not os.path.exists(self.test_norm_fpath):
            self._generate_preprocess_npy_arrays(
                self.train_dpath, 
                self.train_fpath, 
                self.train_norm_fpath
            )
            self._generate_preprocess_npy_arrays(
                self.dev_dpath, 
                self.val_fpath, 
                self.val_norm_fpath
            )
            self._generate_preprocess_npy_arrays(
                self.test_dpath, 
                self.test_fpath, 
                self.test_norm_fpath
            )
        
        # Check if indices Numpy arrays exist
        if not os.path.exists(self.train_idxs_fpath) or \
            not os.path.exists(self.val_idxs_fpath) or \
            not os.path.exists(self.test_idxs_fpath):
            self._generate_idxs()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            X_train_fpath = self.train_norm_fpath if self.normalize else self.train_fpath
            X_val_fpath = self.val_norm_fpath if self.normalize else self.val_fpath
            X_test_fpath = self.test_norm_fpath if self.normalize else self.test_fpath

            self.setup_train = PretrainMaskKeypointDataset(
                self.train_idxs_fpath, 
                X_train_fpath, 
                self.R, 
                self.m, 
                self.K, 
                self.max_disturbance
            )
            self.setup_val = PretrainMaskKeypointDataset(
                self.val_idxs_fpath,
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
    
    def _generate_idxs(self):
        train_idxs = np.arange(len(np.load(self.train_fpath)))
        val_idxs = np.arange(
            start=len(train_idxs), 
            stop=len(train_idxs) + len(np.load(self.val_fpath))
        )
        test_idxs = np.arange(
            start=len(train_idxs) + len(val_idxs),
            stop=len(train_idxs) + len(val_idxs) + len(np.load(self.test_fpath))
        )
        np.save(self.train_idxs_fpath, train_idxs)
        np.save(self.val_idxs_fpath, val_idxs)
        np.save(self.test_idxs_fpath, test_idxs)
    
    def _generate_train_means_stds(self):
        """Compute mean and standard deviation for all x and y coordinates."""
        npy_files = glob.glob(os.path.join(self.train_dpath, '*.npy'))
        npy_concats = np.concatenate([np.load(f)[...,:2] for f in npy_files])
        means = np.mean(npy_concats, axis=(0,1))
        stds = np.std(npy_concats, axis=(0,1))
        np.save(self.means_fpath, means)
        np.save(self.stds_fpath, stds)

    def _generate_preprocess_npy_arrays(self, dpath, out_fpath, norm_out_fpath):
        """
        Process and save sequences of data in numpy format.

        Loads raw sequences from disk, normalizes them by mean and standard 
        deviation, pads them to a uniform length, and finally saves the 
        processed sequences to disk.

        Parameters:
        dpath (str): Directory path where raw sequences are stored.
        out_fpath (str): File path for saving the processed (but not normalized) sequences.
        norm_out_fpath (str): File path for saving the normalized sequences.
        """
        # Load raw sequences from the specified directory
        seqs = self._load_raw_seqs(dpath)
        # Normalize the sequences by their mean and standard deviation
        seqs_norm = self._normalize_seqs(seqs)
        # Pad the sequences to ensure they all have the same length
        seqs = self._pad_seqs_by_max_len(seqs)
        seqs_norm = self._pad_seqs_by_max_len(seqs_norm)
        # Save the processed and normalized sequences to disk
        np.save(out_fpath, seqs)
        np.save(norm_out_fpath, seqs_norm)
        # Free up memory by deleting the large sequence variables and invoking garbage collection
        del seqs
        del seqs_norm
        gc.collect()
 
    def _load_raw_seqs(self, dpath):
        """
        Load raw sequences from .npy files in a specified directory.

        This function scans a given directory for .npy files and loads the data from
        each file into a list. 

        Parameters:
        dpath (str): The directory path where .npy files are stored.

        Returns:
        list: A list of numpy arrays, each array loaded from a .npy file.
        """
        # Use glob to find all .npy files in the specified directory
        npy_files = glob.glob(os.path.join(dpath, '*.npy'))
        # Load each .npy file and append its contents to the list
        seqs = [np.load(f) for f in npy_files]

        return seqs

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
        means = np.load(self.means_fpath)
        stds = np.load(self.stds_fpath)
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
                constant_values=RwthPhoenixDataModule.SEQ_PAD_VALUE
            ) 
            for i, t in enumerate(seqs)
        ])

        return seqs

if __name__ == '__main__':

    d = RwthPhoenixDataModule(
        32,
        True,
        phoenix_T=False
    )
    d.prepare_data()
    d.setup()
    dl = d.train_dataloader()
    sample = next(iter(dl))
    embed()