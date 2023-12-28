import os
import argparse

import torch
import numpy as np
from IPython import embed

from signbert.data_modules.HANDS17DataModule import HANDS17DataModule

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=0.3)
parser.add_argument("--m", type=float, default=5)
parser.add_argument("--K", type=float, default=8)
parser.add_argument("--max_disturbance", type=float, default=0.25)
parser.add_argument("--no_mask_joint", action="store_true")
args = parser.parse_args()

# Create dataset
dataset = HANDS17DataModule(
    batch_size=1, 
    normalize=True,
    R=args.R,
    m=args.m,
    K=args.K,
    max_disturbance=args.max_disturbance,
    no_mask_joint=args.no_mask_joint
)
dataset.prepare_data()
dataset.setup()
dl_train = dataset.train_dataloader()
dl_val = dataset.val_dataloader()
means = torch.from_numpy(np.load(HANDS17DataModule.MEANS_NPY_FPATH)).squeeze()
stds = torch.from_numpy(np.load(HANDS17DataModule.STDS_NPY_FPATH)).squeeze()

# Iterate over dataloaders and compute P@20
p_20_input_train = []
for batch in dl_train:
    seq_original = batch[1]
    seq_masked = batch[2]
    pad_idxs = (seq_original == 0.0).all(3).all(2)
    seq_original = seq_original[torch.logical_not(pad_idxs)]
    seq_masked = seq_masked[torch.logical_not(pad_idxs)]
    seq_original = (seq_original * stds) + means
    # Masked is normalized too, but 0. values need to be preserved
    seq_masked_is_zero = seq_masked == 0.0
    seq_masked = (seq_masked * stds) + means
    seq_masked[seq_masked_is_zero] = 0.0
    input_dists = torch.linalg.norm(seq_original - seq_masked, dim=2)
    p_20_input = (input_dists <= 20.).sum() / input_dists.numel()
    p_20_input_train.append(p_20_input.item())
p_20_input_val = []
for batch in dl_val:
    seq_original = batch[1]
    seq_masked = batch[2]
    pad_idxs = (seq_original == 0.0).all(3).all(2)
    seq_original = seq_original[torch.logical_not(pad_idxs)]
    seq_masked = seq_masked[torch.logical_not(pad_idxs)]
    seq_original = (seq_original * stds) + means
    # Masked is normalized too, but 0. values need to be preserved
    seq_masked_is_zero = seq_masked == 0.0
    seq_masked = (seq_masked * stds) + means
    seq_masked[seq_masked_is_zero] = 0.0
    input_dists = torch.linalg.norm(seq_original - seq_masked, dim=2)
    p_20_input = (input_dists <= 20.).sum() / input_dists.numel()
    p_20_input_val.append(p_20_input.item())

mean_p_20_input_train = sum(p_20_input_train) / len(p_20_input_train)
mean_p_20_input_val = sum(p_20_input_val) / len(p_20_input_val)
print(f"Train mean P@20={mean_p_20_input_train}")
print(f"Validation mean P@20={mean_p_20_input_val}")

# Write to file
out_fpath = "measure_input_p_at_20.txt"
with open(out_fpath, "a+") as fid:
    fid.write(str(args.__dict__))
    fid.write("\n")
    fid.write(f"Train mean P@20={mean_p_20_input_train}")
    fid.write("\n")
    fid.write(f"Validation mean P@20={mean_p_20_input_val}")
    fid.write("\n")
    fid.write("\n")