import os

import torch
import numpy as np
import lightning.pytorch as pl

from signbert.utils import my_import
from signbert.model.PositionalEncoding import PositionalEncoding
from signbert.metrics.PCK import PCK, PCKAUC
from manotorch.manolayer import ManoLayer, MANOOutput
from IPython import embed; from sys import exit


class SignBertModel(pl.LightningModule):
    """
    A PyTorch Lightning module implementing SignBERT for sign language recognition. 

    This class combines gesture extraction, positional encoding, spatial-temporal processing, 
    transformer encoders, and MANO layers to process and interpret sign language gestures.
    
    It handles two-handed configuration used during pre-training.

    Attributes:
    Various configuration parameters like in_channels, num_hid, num_heads, etc.
    ge (GestureExtractor): Module for gesture feature extraction.
    pe (PositionalEncoding): Module for adding positional encoding.
    stpe (SpatialTemporalProcessing): Module for spatial-temporal processing of keypoints.
    te (TransformerEncoder): Transformer encoder for sequence processing.
    pg (Linear): Linear layer for prediction.
    rhand_hd, lhand_hd (ManoLayer): MANO layers for detailed hand pose estimation.
    PCK and PCKAUC metrics for training and validation.
    """
    def __init__(
            self, 
            in_channels, 
            num_hid, 
            num_heads,
            tformer_n_layers,
            tformer_dropout,
            eps, 
            lmbd, 
            weight_beta, 
            weight_delta,
            lr,
            hand_cluster,
            n_pca_components,
            gesture_extractor_cls,
            gesture_extractor_args,
            arms_extractor_cls,
            arms_extractor_args,
            total_steps=None,
            normalize_inputs=False,
            use_pca=True,
            flat_hand=False,
            weight_decay=0.01,
            use_onecycle_lr=False,
            pct_start=None,
            *args,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters()
        # Automatic optimization is disabled so each batch (containing just 
        # examples from one dataset) can be backpropagated independently, so it
        # has to be done manually
        self.automatic_optimization = False

        self.in_channels = in_channels
        self.num_hid = num_hid
        self.num_heads = num_heads
        self.tformer_n_layers = tformer_n_layers
        self.tformer_dropout = tformer_dropout
        self.eps = eps
        self.lmbd = lmbd
        self.weight_beta = weight_beta
        self.weight_delta = weight_delta
        self.total_steps = total_steps
        self.lr = lr
        self.hand_cluster = hand_cluster
        self.n_pca_components = n_pca_components
        self.gesture_extractor_cls = my_import(gesture_extractor_cls)
        self.gesture_extractor_args = gesture_extractor_args
        self.arms_extractor_cls = my_import(arms_extractor_cls)
        self.arms_extractor_args = arms_extractor_args
        self.normalize_inputs = normalize_inputs
        self.use_pca = use_pca
        self.flat_hand = flat_hand
        self.weight_decay = weight_decay
        self.use_onecycle_lr = use_onecycle_lr
        self.pct_start = pct_start
        # Variable to control the input channels dynamically based if clustering is enabled
        num_hid_mult = 1 if hand_cluster else 21
        # Initialization of various components of the model
        self.ge = self.gesture_extractor_cls(**gesture_extractor_args)
        self.pe = PositionalEncoding(
            d_model=num_hid*num_hid_mult,
            dropout=0.1,
            max_len=1000,
        )
        self.stpe = self.arms_extractor_cls(**arms_extractor_args)
        el = torch.nn.TransformerEncoderLayer(d_model=num_hid*num_hid_mult, nhead=num_heads, batch_first=True, dropout=tformer_dropout)
        self.te = torch.nn.TransformerEncoder(el, num_layers=tformer_n_layers)
        self.pg = torch.nn.Linear(
            in_features=num_hid*num_hid_mult,
            out_features=(
                n_pca_components + 3 + # theta + global pose
                10 + # beta
                9 + # rotation matrix
                2 + # translation vector
                1 # scale scalar
            )
        )
        # MANO initalization for both right and left hands
        mano_assets_root = os.path.split(__file__)[0]
        mano_assets_root = os.path.join(mano_assets_root, "thirdparty", "mano_assets")
        assert os.path.isdir(mano_assets_root), "Download MANO files, check README."
        self.rhand_hd = ManoLayer(
            center_idx=0,
            flat_hand_mean=flat_hand,
            mano_assets_root=mano_assets_root,
            use_pca=use_pca,
            ncomps=n_pca_components,
        )
        self.lhand_hd = ManoLayer(
            center_idx=0,
            flat_hand_mean=flat_hand,
            mano_assets_root=mano_assets_root,
            use_pca=use_pca,
            ncomps=n_pca_components,
            side="left"
        )
        # PCK and PCKAUC metrics for training and validation
        self.train_pck_20 = PCK(thr=20)
        self.train_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        self.val_pck_20 = PCK(thr=20)
        self.val_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        # Placeholders
        self.mean_loss = []
        self.mean_pck_20 = []

    def forward(self, arms, rhand, lhand):
        # Concatenate right and left hand data
        x = torch.concat((rhand, lhand), dim=2)
        # Extract hand tokens using gesture extractor
        rhand, lhand = self.ge(x)
        rhand = rhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        lhand = lhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        # Extract arm tokens using spatial-temporal arm extractor
        rarm, larm = self.stpe(arms)
        rarm = rarm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        larm = larm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        N, T, C, V = rhand.shape
        # Combine hands tokens with spatio-temporal positional tokens
        rhand = rhand + rarm 
        lhand = lhand + larm 
        # Reshape hand data for processing
        rhand = rhand.view(N, T, C*V)
        lhand = lhand.view(N, T, C*V)
        # Apply positional encoding
        rhand = self.pe(rhand) 
        lhand = self.pe(lhand) 
        # Process data through the transformer encoder
        rhand = self.te(rhand)
        lhand = self.te(lhand)
        # Predict hand and camera parameters for right and left hands
        rhand_params = self.pg(rhand)
        lhand_params = self.pg(lhand)
        # Extract both hands parameters
        offset = self.n_pca_components + 3
        rhand_pose_coeffs = rhand_params[...,:offset]
        rhand_betas = rhand_params[...,offset:offset+10]
        lhand_pose_coeffs = lhand_params[...,:offset]
        lhand_betas = lhand_params[...,offset:offset+10]
        offset += 10
        rhand_R = rhand_params[...,offset:offset+9]
        rhand_R = rhand_R.view(N, T, 3, 3)
        lhand_R = lhand_params[...,offset:offset+9]
        lhand_R = lhand_R.view(N, T, 3, 3)
        offset +=9
        rhand_O = rhand_params[...,offset:offset+2]
        lhand_O = lhand_params[...,offset:offset+2]
        offset += 2
        rhand_S = rhand_params[...,offset:offset+1]
        lhand_S = lhand_params[...,offset:offset+1]
        rhand_pose_coeffs = rhand_pose_coeffs.view(N*T, -1)
        lhand_pose_coeffs = lhand_pose_coeffs.view(N*T, -1)
        rhand_betas = rhand_betas.view(N*T, -1)
        lhand_betas = lhand_betas.view(N*T, -1)
        # Apply the MANO model to obtain 3D joints and vertices
        rhand_mano_output: MANOOutput = self.rhand_hd(rhand_pose_coeffs, rhand_betas)
        lhand_mano_output: MANOOutput = self.lhand_hd(lhand_pose_coeffs, lhand_betas)
        # Extract and reshape the MANO output for both hands
        rhand_vertices = rhand_mano_output.verts
        rhand_joints_3d = rhand_mano_output.joints
        rhand_pose_coeffs = rhand_pose_coeffs.view(N, T, -1)
        rhand_betas = rhand_betas.view(N, T, -1)
        rhand_vertices = rhand_vertices.view(N, T, 778, 3).detach().cpu()
        rhand_center_joint = rhand_mano_output.center_joint.detach().cpu()
        rhand_joints_3d = rhand_joints_3d.view(N, T, 21, 3)
        lhand_vertices = lhand_mano_output.verts
        lhand_joints_3d = lhand_mano_output.joints
        lhand_pose_coeffs = lhand_pose_coeffs.view(N, T, -1)
        lhand_betas = lhand_betas.view(N, T, -1)
        lhand_vertices = lhand_vertices.view(N, T, 778, 3).detach().cpu()
        lhand_center_joint = lhand_mano_output.center_joint.detach().cpu()
        lhand_joints_3d = lhand_joints_3d.view(N, T, 21, 3)
        # Apply ortographic projection to the 3D joints to obtain 2D image coordinates
        rhand = torch.matmul(rhand_R, rhand_joints_3d.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        rhand = rhand[...,:2]
        rhand *= rhand_S.unsqueeze(-1)
        rhand += rhand_O.unsqueeze(2)
        lhand = torch.matmul(lhand_R, lhand_joints_3d.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        lhand = lhand[...,:2]
        lhand *= lhand_S.unsqueeze(-1)
        lhand += lhand_O.unsqueeze(2)
        # Return processed data
        return {
            "rhand": (rhand, rhand_pose_coeffs, rhand_betas, rhand_vertices, rhand_R, rhand_S, rhand_O, rhand_center_joint, rhand_joints_3d),
            "lhand": (lhand, lhand_pose_coeffs, lhand_betas, lhand_vertices, lhand_R, lhand_S, lhand_O, lhand_center_joint, lhand_joints_3d)
        }

    def training_step(self, batch, batch_idx):
        # Get optimizer and scheduler (part of the manual optimization)
        opt = self.optimizers()
        sch = self.lr_schedulers()
        # Process <key-value> pairs in the batch (<dataset_name:batch_data>)
        for k, v in batch.items():
            # Unpack the batch data
            (seq_idx, 
            arms,
            rhand, 
            rhand_masked,
            rhand_masked_frames_idx,
            rhand_scores,
            lhand, 
            lhand_masked,
            lhand_masked_frames_idx,
            lhand_scores) = v
            # Forward pass through the model
            hand_data = self(arms, rhand_masked, lhand_masked)
            # Extract logits, pose coefficients, and betas from the model's output
            (rhand_logits, rhand_theta, rhand_beta, _, _, _, _, _, _) = hand_data["rhand"]
            (lhand_logits, lhand_theta, lhand_beta, _, _, _, _, _, _) = hand_data["lhand"]
            # Loss only applied on frames with masked joints
            rhand_valid_idxs = torch.where(rhand_masked_frames_idx != -1.)
            rhand_logits = rhand_logits[rhand_valid_idxs]
            rhand = rhand[rhand_valid_idxs]
            rhand_scores = rhand_scores[rhand_valid_idxs]
            lhand_valid_idxs = torch.where(lhand_masked_frames_idx != -1.)
            lhand_logits = lhand_logits[lhand_valid_idxs]
            lhand = lhand[lhand_valid_idxs]
            lhand_scores = lhand_scores[lhand_valid_idxs]
            # Compute reconstruction loss (LRec) and regularization loss (LReg) for both hands
            rhand_lrec = torch.norm(rhand_logits - rhand, p=1, dim=2)
            rhand_scores = torch.where(rhand_scores >= self.eps, 1., rhand_scores)
            rhand_lrec = (rhand_lrec * rhand_scores).sum()
            lhand_lrec = torch.norm(lhand_logits - lhand, p=1, dim=2)
            lhand_scores = torch.where(lhand_scores >= self.eps, 1., lhand_scores)
            lhand_lrec = (lhand_lrec * lhand_scores).sum()
            rhand_beta_t_minus_one = torch.roll(rhand_beta, shifts=1, dims=1)
            rhand_beta_t_minus_one[:, 0] = 0.
            rhand_lreg = torch.norm(rhand_theta, 2) + self.weight_beta * torch.norm(rhand_beta, 2) + \
                self.weight_delta * torch.norm(rhand_beta - rhand_beta_t_minus_one, 2)
            # Combine right hand LRec and LReg
            rhand_loss = rhand_lrec + (self.lmbd * rhand_lreg)
            lhand_lrec = torch.norm(lhand_logits[lhand_scores>self.eps] - lhand[lhand_scores>self.eps], p=1, dim=1).sum()
            lhand_beta_t_minus_one = torch.roll(lhand_beta, shifts=1, dims=1)
            lhand_beta_t_minus_one[:, 0] = 0.
            lhand_lreg = torch.norm(lhand_theta, 2) + self.weight_beta * torch.norm(lhand_beta, 2) + \
                self.weight_delta * torch.norm(lhand_beta - lhand_beta_t_minus_one, 2)
            # Combine left hand LRec and LReg
            lhand_loss = lhand_lrec + (self.lmbd * lhand_lreg)
            # Combine both losses
            loss = rhand_loss + lhand_loss
            # Manual backward pass
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            if isinstance(sch, torch.optim.lr_scheduler.OneCycleLR):
                sch.step()
            if self.normalize_inputs:
                # Check that means and stds are in the same device as trainer
                if self.device != self.trainer.datamodule.means[k].device:
                   self.trainer.datamodule.means[k] = self.trainer.datamodule.means[k].to(self.device)
                if self.device != self.trainer.datamodule.stds[k].device:
                    self.trainer.datamodule.stds[k] = self.trainer.datamodule.stds[k].to(self.device)
                # Reverse mean 0 and std 1 normalization to obtain 2D image coordinates
                means = self.trainer.datamodule.means[k]
                stds = self.trainer.datamodule.stds[k]
                rhand_logits = (rhand_logits * stds) + means
                lhand_logits = (lhand_logits * stds) + means
                rhand = (rhand * stds) + means
                lhand = (lhand * stds) + means
            # Compute PCK (Percentage of Correct Keypoints) metrics
            rhand_pck_20 = self.train_pck_20.update(preds=rhand_logits, target=rhand)
            rhand_pck_20 = self.train_pck_20.compute()
            self.train_pck_20.reset()
            lhand_pck_20 = self.train_pck_20(preds=lhand_logits, target=lhand)
            lhand_pck_20 = self.train_pck_20.compute()
            self.train_pck_20.reset()
            rhand_pck_auc_20_40 = self.train_pck_auc_20_40(preds=rhand_logits, target=rhand)
            rhand_pck_auc_20_40 = self.train_pck_auc_20_40.compute()
            self.train_pck_auc_20_40.reset()
            lhand_pck_auc_20_40 = self.train_pck_auc_20_40(preds=lhand_logits, target=lhand)
            lhand_pck_auc_20_40 = self.train_pck_auc_20_40.compute()
            self.train_pck_auc_20_40.reset()
            # Log metrics
            self.log(f"{k}_train_loss", loss, prog_bar=False)
            self.log(f"{k}_train_rhand_PCK_20", rhand_pck_20, prog_bar=False)
            self.log(f"{k}_train_lhand_PCK_20", lhand_pck_20, prog_bar=False)
            self.log(f"{k}_train_rhand_PCK_auc_20_40", rhand_pck_auc_20_40, prog_bar=False)
            self.log(f"{k}_train_lhand_PCK_auc_20_40", lhand_pck_auc_20_40, prog_bar=False)
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        # Identify the dataset key based on the dataloader index
        dataset_key = list(self.trainer.datamodule.val_dataloaders.keys())[dataloader_idx]
        # Unpack the batch data
        (seq_idx, 
        arms,
        rhand, 
        rhand_masked,
        rhand_masked_frames_idx,
        rhand_scores,
        lhand, 
        lhand_masked,
        lhand_masked_frames_idx,
        lhand_scores) = batch
        # Process data through the model
        hand_data = self(arms, rhand_masked, lhand_masked)
        # Extract relevant outputs from the model for both right and left hands
        (rhand_logits, rhand_theta, rhand_beta, _, _, _, _, _, _) = hand_data["rhand"]
        (lhand_logits, lhand_theta, lhand_beta, _, _, _, _, _, _) = hand_data["lhand"]
        # Compute loss for both hands based on the masked frames
        rhand_valid_idxs = torch.where(rhand_masked_frames_idx != -1.)
        rhand_logits = rhand_logits[rhand_valid_idxs]
        rhand = rhand[rhand_valid_idxs]
        rhand_scores = rhand_scores[rhand_valid_idxs]
        lhand_valid_idxs = torch.where(lhand_masked_frames_idx != -1.)
        lhand_logits = lhand_logits[lhand_valid_idxs]
        lhand = lhand[lhand_valid_idxs]
        lhand_scores = lhand_scores[lhand_valid_idxs]
        # Compute LRec and LReg
        rhand_lrec = torch.norm(rhand_logits - rhand, p=1, dim=2)
        rhand_scores = torch.where(rhand_scores >= self.eps, 1., rhand_scores)
        rhand_lrec = (rhand_lrec * rhand_scores).sum()
        lhand_lrec = torch.norm(lhand_logits - lhand, p=1, dim=2)
        lhand_scores = torch.where(lhand_scores >= self.eps, 1., lhand_scores)
        lhand_lrec = (lhand_lrec * lhand_scores).sum()
        rhand_beta_t_minus_one = torch.roll(rhand_beta, shifts=1, dims=1)
        rhand_beta_t_minus_one[:, 0] = 0.
        rhand_lreg = torch.norm(rhand_theta, 2) + self.weight_beta * torch.norm(rhand_beta, 2) + \
            self.weight_delta * torch.norm(rhand_beta - rhand_beta_t_minus_one, 2)
        rhand_loss = rhand_lrec + (self.lmbd * rhand_lreg)
        lhand_lrec = torch.norm(lhand_logits[lhand_scores>self.eps] - lhand[lhand_scores>self.eps], p=1, dim=1).sum()
        lhand_beta_t_minus_one = torch.roll(lhand_beta, shifts=1, dims=1)
        lhand_beta_t_minus_one[:, 0] = 0.
        lhand_lreg = torch.norm(lhand_theta, 2) + self.weight_beta * torch.norm(lhand_beta, 2) + \
            self.weight_delta * torch.norm(lhand_beta - lhand_beta_t_minus_one, 2)
        lhand_loss = lhand_lrec + (self.lmbd * lhand_lreg)
        # Combine both losses
        loss = rhand_loss + lhand_loss
        # Compute metrics
        if self.normalize_inputs:
            # Check that means and stds are in the same device as trainer
            if self.device != self.trainer.datamodule.means[dataset_key].device:
                self.trainer.datamodule.means[dataset_key] = self.trainer.datamodule.means[dataset_key].to(self.device)
            if self.device != self.trainer.datamodule.stds[dataset_key].device:
                self.trainer.datamodule.stds[dataset_key] = self.trainer.datamodule.stds[dataset_key].to(self.device)
            # Reverse normalization
            means = self.trainer.datamodule.means[dataset_key]
            stds = self.trainer.datamodule.stds[dataset_key]
            rhand_logits = (rhand_logits * stds) + means
            lhand_logits = (lhand_logits * stds) + means
            rhand = (rhand * stds) + means
            lhand = (lhand * stds) + means
        # Compute metrics   
        rhand_pck_20 = self.val_pck_20.update(preds=rhand_logits, target=rhand)
        rhand_pck_20 = self.val_pck_20.compute()
        self.val_pck_20.reset()
        lhand_pck_20 = self.val_pck_20(preds=lhand_logits, target=lhand)
        lhand_pck_20 = self.val_pck_20.compute()
        self.val_pck_20.reset()
        rhand_pck_auc_20_40 = self.val_pck_auc_20_40(preds=rhand_logits, target=rhand)
        rhand_pck_auc_20_40 = self.val_pck_auc_20_40.compute()
        self.val_pck_auc_20_40.reset()
        lhand_pck_auc_20_40 = self.val_pck_auc_20_40(preds=lhand_logits, target=lhand)
        lhand_pck_auc_20_40 = self.val_pck_auc_20_40.compute()
        self.val_pck_auc_20_40.reset()
        # Log metrics
        self.log(f"{dataset_key}_val_loss", loss, prog_bar=False)
        self.log(f"{dataset_key}_val_rhand_pck_20", rhand_pck_20, prog_bar=False)
        self.log(f"{dataset_key}_val_lhand_pck_20", lhand_pck_20, prog_bar=False)
        self.log(f"{dataset_key}_val_rhand_pck_auc_20_40", rhand_pck_auc_20_40, prog_bar=False)
        self.log(f"{dataset_key}_val_lhand_pck_auc_20_40", lhand_pck_auc_20_40, prog_bar=False)
        # Store epoch level average results
        self.mean_loss.append(loss.detach().cpu())
        self.mean_pck_20.append(rhand_pck_20)
        self.mean_pck_20.append(lhand_pck_20)
    
    def on_validation_epoch_end(self):
        # Compute the mean of the metrics at the end of the epoch
        mean_epoch_loss = torch.stack(self.mean_loss).mean()
        mean_epoch_pck_20 = torch.stack(self.mean_pck_20).mean()
        self.log("val_loss", mean_epoch_loss, prog_bar=False)
        self.log("val_PCK_20", mean_epoch_pck_20, prog_bar=False)
        self.mean_loss.clear()
        self.mean_pck_20.clear()

    def configure_optimizers(self):
        toret = {}
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.use_onecycle_lr:
            lr_scheduler_config = dict(
                scheduler=torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=self.lr,
                    total_steps=self.trainer.estimated_stepping_batches,
                    pct_start=self.pct_start,
                    anneal_strategy='linear'
                )
            )
        toret['optimizer'] = optimizer
        if self.use_onecycle_lr:
            toret['lr_scheduler'] = lr_scheduler_config

        return toret