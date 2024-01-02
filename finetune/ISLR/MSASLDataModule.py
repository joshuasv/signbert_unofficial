import os
import re
from glob import glob

import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from signbert.utils import read_json, read_txt_as_list

from IPython import embed;


class MSASLDataModule(pl.LightningDataModule):
    DPATH = '/home/tmpvideos/SLR/MSASL'
    PREPROCESS_DPATH = os.path.join(DPATH, 'preprocess')
    MISSING_VIDEOS_FPATH = os.path.join(DPATH, 'raw_videos', 'missing.txt')
    SKELETON_DPATH = os.path.join(DPATH, 'skeleton-data', 'rtmpose-l_8xb64-270e_coco-wholebody-256x192')
    TRAIN_SPLIT_JSON_FPATH = os.path.join(DPATH, 'MSASL_train.json')
    VAL_SPLIT_JSON_FPATH = os.path.join(DPATH, 'MSASL_val.json')
    TEST_SPLIT_JSON_FPATH = os.path.join(DPATH, 'MSASL_test.json')
    CLASSES_JSON_FPATH = os.path.join(DPATH, 'MSASL_classes.json')
    TRAIN_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'train')
    VAL_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'val')
    TEST_SKELETON_DPATH = os.path.join(SKELETON_DPATH, 'test')
    MEANS_FPATH = os.path.join(PREPROCESS_DPATH, 'means.npy')
    STDS_FPATH = os.path.join(PREPROCESS_DPATH, 'stds.npy')
    VIDEO_ID_PATTERN = r"(?<=v\=).{11}"
    PADDING_VALUE = 0.0

    def __init__(self, batch_size, normalize):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize

    def setup(self, stage):
        classes = read_json(MSASLDataModule.CLASSES_JSON_FPATH)
        missing_video_ids = read_txt_as_list(MSASLDataModule.MISSING_VIDEOS_FPATH) 
        if stage == "fit":
            train_info = read_json(MSASLDataModule.TRAIN_SPLIT_JSON_FPATH)
            # Add class id 
            [ti.update(class_id=classes.index(ti["text"])) for ti in train_info]
            # Add video id
            [ti.update(video_id=re.search(MSASLDataModule.VIDEO_ID_PATTERN, ti["url"]).group()) for ti in train_info]
            # Filter missing videos
            train_info = [ti for ti in train_info if ti["video_id"] not in missing_video_ids]
            self.train_dataset = MSASLDataset(
                train_info, 
                MSASLDataModule.TRAIN_SKELETON_DPATH, 
                self.normalize,
                np.load(MSASLDataModule.MEANS_FPATH),
                np.load(MSASLDataModule.STDS_FPATH)
            )
            val_info = read_json(MSASLDataModule.VAL_SPLIT_JSON_FPATH)
            # Add class id 
            [ti.update(class_id=classes.index(ti["text"])) for ti in val_info]
            # Add video id
            [ti.update(video_id=re.search(MSASLDataModule.VIDEO_ID_PATTERN, ti["url"]).group()) for ti in val_info]
            # Filter missing videos
            val_info = [ti for ti in val_info if ti["video_id"] not in missing_video_ids]
            self.val_dataset = MSASLDataset(
                val_info, 
                MSASLDataModule.VAL_SKELETON_DPATH, 
                self.normalize,
                np.load(MSASLDataModule.MEANS_FPATH),
                np.load(MSASLDataModule.STDS_FPATH)
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True,
            collate_fn=my_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            collate_fn=my_collate_fn
        )


class MSASLDataset(Dataset):

    def __init__(self, train_info, skeleton_dpath, normalize, normalize_mean=None, normalize_std=None):
        super().__init__()
        self.train_info = train_info
        self.skeleton_dpath = skeleton_dpath
        self.normalize = normalize
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self):
        return len(self.train_info)

    def __getitem__(self, idx):
        sample = self.train_info[idx]
        class_id = sample["class_id"]
        video_id = sample["video_id"]
        start_video = sample["start"]
        end_video = sample["end"]
        skeleton_video_fpath = os.path.join(self.skeleton_dpath, f"{video_id}.npy")
        skeleton_data = np.load(skeleton_video_fpath)[start_video:end_video]
        # Drop score
        skeleton_data = skeleton_data[...,:2]
        if self.normalize:
            skeleton_data = (skeleton_data - self.normalize_mean) / self.normalize_std
        arms = skeleton_data[:, 5:11]
        lhand = skeleton_data[:, 91:112]
        rhand = skeleton_data[:, 112:133]

        return {
            "sample_id": idx,
            "class_id": class_id,
            "arms": arms, 
            "lhand": lhand,
            "rhand": rhand
        }


def my_collate_fn(original_batch):
    sample_id = []
    class_id = []
    arms = []
    lhand = []
    rhand = []
    for ob in original_batch:
        sample_id.append(ob["sample_id"])
        class_id.append(ob["class_id"])
        arms.append(torch.from_numpy(ob["arms"]))
        lhand.append(torch.from_numpy(ob["lhand"]))
        rhand.append(torch.from_numpy(ob["rhand"]))
    arms = pad_sequence(arms, batch_first=True, padding_value=MSASLDataModule.PADDING_VALUE)
    lhand = pad_sequence(lhand, batch_first=True, padding_value=MSASLDataModule.PADDING_VALUE)
    rhand = pad_sequence(rhand, batch_first=True, padding_value=MSASLDataModule.PADDING_VALUE)
    class_id = torch.tensor(class_id, dtype=torch.int64)
    sample_id = torch.tensor(sample_id, dtype=torch.int32)

    return {
        "sample_id": sample_id,
        "class_id": class_id,
        "arms": arms, 
        "lhand": lhand,
        "rhand": rhand
    }


if __name__ == "__main__":

    d = MSASLDataModule(batch_size=32, normalize=True)
    d.setup(stage="fit")
    dl = d.train_dataloader()
    batch = next(iter(dl))
    embed(); exit()


