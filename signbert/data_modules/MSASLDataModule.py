import os
import gc
import glob

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from signbert.data_modules.MaskKeypointDataset import MaskKeypointDataset, mask_keypoint_dataset_collate_fn
from signbert.utils import read_json

from IPython import embed


class MSASLDataModule(pl.LightningDataModule):

    DPATH = '/home/gts/projects/jsoutelo/SignBERT+/datasets/WLASL/start_kit'
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

            self.setup_train = MaskKeypointDataset(
                WLASLDataModule.TRAIN_IDXS_FPATH, 
                X_train_fpath, 
                self.R, 
                self.m, 
                self.K, 
                self.max_disturbance
            )
            self.setup_val = MaskKeypointDataset(
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

    def _populate_video_id_by_split(self, data_json):
        train = []
        val = []
        test = []
        for d in data_json:
            instances = d['instances']
            for i in instances:
                video_id = i['video_id']
                split = i['split']
                if split == 'train':
                    train.append(video_id)
                elif split == 'val':
                    val.append(video_id)
                elif split == 'test':
                    test.append(video_id)
                else:
                    raise ValueError(f'split {split} not recognized')
        
        return train, val, test

    def _load_data_by_split(self, split_idxs, skeleton_fpaths):
        data = []
        idxs = []
        for idx in split_idxs:
            for fpath in skeleton_fpaths:
                fpath_id = os.path.split(fpath)[-1].replace('.npy', '')
                if fpath_id == idx:
                    idxs.append(idx)
                    data.append(np.load(fpath))
                    break

        return data, idxs

    def _generate_means_stds(self, train_data):
        seq_concats = np.concatenate([s[..., :2] for s in train_data], axis=0)
        means = seq_concats.mean((0, 1))
        stds = seq_concats.std((0, 1))
        np.save(WLASLDataModule.MEANS_FPATH, means)
        np.save(WLASLDataModule.STDS_FPATH, stds)

    def _generate_preprocess_npy_arrays(
            self, 
            split_idxs, 
            skeleton_fpaths, 
            out_fpath,
            norm_out_fpath,
            idxs_out_fpath,
        ):
        seqs, seqs_idxs = self._load_data_by_split(split_idxs, skeleton_fpaths)
        seqs_norm = self._normalize_seqs(seqs)
        seqs = self._pad_seqs_by_max_len(seqs)
        seqs_norm = self._pad_seqs_by_max_len(seqs_norm)
        seqs = seqs.astype(np.float32)
        seqs_norm = seqs_norm.astype(np.float32)
        seqs_idxs = np.array(seqs_idxs, dtype=np.int32)
        np.save(out_fpath, seqs)
        np.save(norm_out_fpath, seqs_norm)
        np.save(idxs_out_fpath, seqs_idxs)
        del seqs
        del seqs_norm
        del seqs_idxs
        gc.collect()
    
    def _normalize_seqs(self, seqs):
        means = np.load(WLASLDataModule.MEANS_FPATH)
        stds = np.load(WLASLDataModule.STDS_FPATH)
        # Append identity to not affect the score
        means = np.concatenate((means, [0]), -1)
        stds = np.concatenate((stds, [1]), -1)
        seqs_norm = [(s - means) / stds for s in seqs]

        return seqs_norm
    
    def _pad_seqs_by_max_len(self, seqs):
        seqs_len = [len(t) for t in seqs]
        max_seq_len = max(seqs_len)
        lmdb_gen_pad_seq = lambda s_len: ((0,max_seq_len-s_len), (0,0), (0,0))
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