import os
import gc
import glob

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from signbert.data_modules.PretrainMaskKeypointDataset import PretrainMaskKeypointDataset, mask_keypoint_dataset_collate_fn
from signbert.utils import read_json

from IPython import embed


class WLASLDataModule(pl.LightningDataModule):

    DPATH = '/home/tmpvideos/SLR/WLASL-raw-data-and-mmpose/start_kit/'
    SPLIT_DATA_JSON_FPAHT = os.path.join(DPATH, 'WLASL_v0.3.json')
    SKELETON_DPAHT = os.path.join(DPATH, 'skeleton-data', 'rtmpose-l_8xb64-270e_coco-wholebody-256x192')
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
    SEQ_PAD_VALUE = 0.0

    def __init__(self, batch_size, normalize=False, R=0.3, m=5, K=8, max_disturbance=0.25):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.R = R
        self.m = m
        self.K = K
        self.max_disturbance = max_disturbance
        self.means_fpath = WLASLDataModule.MEANS_FPATH
        self.stds_fpath = WLASLDataModule.STDS_FPATH

    def prepare_data(self):
        # Create preprocess path if it does not exist
        if not os.path.exists(WLASLDataModule.PREPROCESS_DPATH):
            os.makedirs(WLASLDataModule.PREPROCESS_DPATH)

        # Compute means and stds
        if not os.path.exists(WLASLDataModule.MEANS_FPATH) or \
            not os.path.exists(WLASLDataModule.STDS_FPATH):
            # Grab train, validation, and test splits data
            splits_data = read_json(WLASLDataModule.SPLIT_DATA_JSON_FPAHT)
            # Associate video_id with split
            train_idxs, val_idxs, test_idxs = self._populate_video_id_by_split(
                splits_data
            )
            # Grab skeleton file paths
            skeleton_fpaths = glob.glob(
                os.path.join(WLASLDataModule.SKELETON_DPAHT, '*.npy')
            )
            # Load data
            train, train_idxs = self._load_data_by_split(train_idxs, skeleton_fpaths)
            self._generate_means_stds(train)
        
        if not os.path.exists(WLASLDataModule.TRAIN_FPATH) or \
            not os.path.exists(WLASLDataModule.VAL_FPATH) or \
            not os.path.exists(WLASLDataModule.TEST_FPATH) or \
            not os.path.exists(WLASLDataModule.TRAIN_NORM_FPATH) or \
            not os.path.exists(WLASLDataModule.VAL_NORM_FPATH) or \
            not os.path.exists(WLASLDataModule.TEST_NORM_FPATH) or \
            not os.path.exists(WLASLDataModule.TRAIN_IDXS_FPATH) or \
            not os.path.exists(WLASLDataModule.VAL_IDXS_FPATH) or \
            not os.path.exists(WLASLDataModule.TEST_IDXS_FPATH):
            # Grab train, validation, and test splits data
            splits_data = read_json(WLASLDataModule.SPLIT_DATA_JSON_FPAHT)
            # Associate video_id with split
            train_idxs, val_idxs, test_idxs = self._populate_video_id_by_split(
                splits_data
            )
            # Grab skeleton file paths
            skeleton_fpaths = glob.glob(
                os.path.join(WLASLDataModule.SKELETON_DPAHT, '*.npy')
            )
            # Generate Numpy
            self._generate_preprocess_npy_arrays(
                train_idxs, 
                skeleton_fpaths, 
                WLASLDataModule.TRAIN_FPATH, 
                WLASLDataModule.TRAIN_NORM_FPATH,
                WLASLDataModule.TRAIN_IDXS_FPATH
            )
            self._generate_preprocess_npy_arrays(
                val_idxs, 
                skeleton_fpaths, 
                WLASLDataModule.VAL_FPATH, 
                WLASLDataModule.VAL_NORM_FPATH,
                WLASLDataModule.VAL_IDXS_FPATH
            )
            self._generate_preprocess_npy_arrays(
                test_idxs, 
                skeleton_fpaths, 
                WLASLDataModule.TEST_FPATH, 
                WLASLDataModule.TEST_NORM_FPATH,
                WLASLDataModule.TEST_IDXS_FPATH
            )
            
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            X_train_fpath = WLASLDataModule.TRAIN_NORM_FPATH if self.normalize else WLASLDataModule.TRAIN_FPATH
            X_val_fpath = WLASLDataModule.VAL_NORM_FPATH if self.normalize else WLASLDataModule.VAL_FPATH
            X_test_fpath = WLASLDataModule.TEST_NORM_FPATH if self.normalize else WLASLDataModule.TEST_FPATH

            self.setup_train = PretrainMaskKeypointDataset(
                WLASLDataModule.TRAIN_IDXS_FPATH, 
                X_train_fpath, 
                self.R, 
                self.m, 
                self.K, 
                self.max_disturbance
            )
            self.setup_val = PretrainMaskKeypointDataset(
                WLASLDataModule.VAL_IDXS_FPATH,
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

    def segregate_data_splits(data_json):
        """
        Segregate data instances into training, validation, and test sets.

        This function iterates over a JSON-like list of data instances, each containing
        information about video IDs and their respective dataset splits (train, val, test),
        and categorizes these IDs into separate lists for training, validation, and testing.

        Parameters:
        data_json (list): A list of dictionaries, each representing a data instance with 
                        'video_id' and 'split' attributes.

        Returns:
        tuple: Three lists containing video IDs for training, validation, and testing, respectively.
        """
        # Initialize lists to store video IDs for each data split
        train, val, test = [], [], []
        # Iterate over each data instance in the provided JSON
        for d in data_json:
            instances = d['instances']
            for i in instances:
                video_id = i['video_id']
                split = i['split']
                # Categorize the video ID based on the split attribute
                if split == 'train':
                    train.append(video_id)
                elif split == 'val':
                    val.append(video_id)
                elif split == 'test':
                    test.append(video_id)
                else:
                    # Raise an error if an unrecognized split is encountered
                    raise ValueError(f'split {split} not recognized')

        return train, val, test

    def _load_data_by_split(self, split_idxs, skeleton_fpaths):
        """
        Load data from files corresponding to specified indices.

        This function iterates through a list of indices and a list of file paths. For each index,
        it finds the file path with the matching index in its name and loads the data from that file.

        Parameters:
        split_idxs (list): A list of indices used to identify which data to load.
        skeleton_fpaths (list): A list of file paths where the data is stored.

        Returns:
        tuple: A tuple containing two lists:
            - data: The data loaded from the files.
            - idxs: The indices corresponding to the loaded data.
        """
        # Initialize lists to store the loaded data and corresponding indices
        data = []
        idxs = []
        # Iterate over each index in the provided list of split indices
        for idx in split_idxs:
            # Iterate over each file path in the provided list of skeleton file paths
            for fpath in skeleton_fpaths:
                # Extract the file ID from the file path
                fpath_id = os.path.split(fpath)[-1].replace('.npy', '')
                # Check if the file ID matches the current index
                if fpath_id == idx:
                    # If matched, append the index and load the data from the file
                    idxs.append(idx)
                    data.append(np.load(fpath))
                    break  # Exit the inner loop once the matching file is found

        return data, idxs


    def _generate_means_stds(self, train_data):
        """Compute mean and standard deviation for all x and y coordinates."""
        seq_concats = np.concatenate([s[..., :2] for s in train_data], axis=0)
        means = seq_concats.mean((0, 1))
        stds = seq_concats.std((0, 1))
        np.save(WLASLDataModule.MEANS_FPATH, means)
        np.save(WLASLDataModule.STDS_FPATH, stds)

    def process_and_save_data(self, split_idxs, skeleton_fpaths, out_fpath, norm_out_fpath, idxs_out_fpath):
        """
        Process and save sequences of data.

        This function loads sequences based on given indices, normalizes them, pads them to a 
        uniform length, converts them to float32 format, and saves the processed data.

        Parameters:
        split_idxs (list): Indices indicating which sequences to load.
        skeleton_fpaths (list): File paths of the skeleton sequences.
        out_fpath (str): File path for saving the processed sequences.
        norm_out_fpath (str): File path for saving the normalized sequences.
        idxs_out_fpath (str): File path for saving the indices of the sequences.
        """
        # Load sequences and their indices
        seqs, seqs_idxs = self._load_data_by_split(split_idxs, skeleton_fpaths)
        # Normalize the sequences
        seqs_norm = self._normalize_seqs(seqs)
        # Pad the sequences to ensure they all have the same length
        seqs = self._pad_seqs_by_max_len(seqs)
        seqs_norm = self._pad_seqs_by_max_len(seqs_norm)
        # Convert sequences to float32 for compatibility with many ML frameworks
        seqs = seqs.astype(np.float32)
        seqs_norm = seqs_norm.astype(np.float32)
        # Convert the sequence indices to int32
        seqs_idxs = np.array(seqs_idxs, dtype=np.int32)
        # Save the processed sequences and their indices
        np.save(out_fpath, seqs)
        np.save(norm_out_fpath, seqs_norm)
        np.save(idxs_out_fpath, seqs_idxs)
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
        means = np.load(WLASLDataModule.MEANS_FPATH)
        stds = np.load(WLASLDataModule.STDS_FPATH)
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
                constant_values=WLASLDataModule.SEQ_PAD_VALUE
            ) 
            for i, t in enumerate(seqs)
        ])

        return seqs


if __name__ == '__main__':

    d = WLASLDataModule(
        batch_size=32,
        normalize=True,
    )
    d.prepare_data()
    d.setup()
    dl = d.train_dataloader()
    sample = next(iter(dl))
    embed()