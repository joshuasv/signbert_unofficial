import os

import torch
import numpy as np
import lightning.pytorch as pl

from signbert.utils import my_import
from signbert.metrics.PCK import PCK, PCKAUC
from signbert.model.PositionalEncoding import PositionalEncoding
from manotorch.manolayer import ManoLayer, MANOOutput
from IPython import embed; from sys import exit


class SignBertModel(pl.LightningModule):
    """
    A PyTorch Lightning module implementing SignBERT for sign language recognition. 

    This class combines gesture extraction, positional encoding, spatial-temporal processing, 
    transformer encoders, and MANO layers to process and interpret sign language gestures.

    Used for the feasibility study on the HANDS17 dataset.

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
        el = torch.nn.TransformerEncoderLayer(d_model=num_hid*num_hid_mult, nhead=num_heads, batch_first=True, dropout=tformer_dropout)
        self.pe = PositionalEncoding(
            d_model=num_hid*num_hid_mult,
            dropout=0.1,
            max_len=2000,
        )
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
        # Initialization of various components of the model
        mano_assets_root = os.path.split(__file__)[0]
        mano_assets_root = os.path.join(mano_assets_root, "thirdparty", "mano_assets")
        assert os.path.isdir(mano_assets_root), "Download MANO files, check README."
        self.hd = ManoLayer(
            center_idx=0,
            flat_hand_mean=flat_hand,
            mano_assets_root=mano_assets_root,
            use_pca=use_pca,
            ncomps=n_pca_components,
        )
        # PCK and PCKAUC metrics for training and validation
        self.train_pck_20 = PCK(thr=20)
        self.train_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        self.val_pck_20 = PCK(thr=20)
        self.val_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        # Placeholders
        self.train_step_losses = []
        self.val_step_losses = []

    def forward(self, x):
        # Extract hand tokens using gesture extractor
        x = self.ge(x)
        # Remove last dimension M and permute to be (N, T, C, V)
        x = x.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        N, T, C, V = x.shape
        x = x.view(N, T, C*V)
        # Apply positional encoding
        x = self.pe(x)
        # Process data through the transformer encoder
        x = self.te(x)
        # Predict hand and camera parameters 
        params = self.pg(x)
        # Extract hand parameters
        offset = self.n_pca_components + 3
        pose_coeffs = params[...,:offset]
        betas = params[...,offset:offset+10]
        offset += 10
        # Extract camera parameters
        R = params[...,offset:offset+9]
        R = R.view(N, T, 3, 3)
        offset +=9
        O = params[...,offset:offset+2]
        offset += 2
        S = params[...,offset:offset+1]
        # Reshape hand parameters for processing 
        pose_coeffs = pose_coeffs.view(N*T, -1)
        betas = betas.view(N*T, -1)
        # Apply the MANO model to obtain 3D joints and vertices
        mano_output: MANOOutput = self.hd(pose_coeffs, betas)
        # Extract and reshape the MANO output
        vertices = mano_output.verts
        joints_3d = mano_output.joints
        pose_coeffs = pose_coeffs.view(N, T, -1)
        betas = betas.view(N, T, -1)
        vertices = vertices.view(N, T, 778, 3).detach().cpu()
        center_joint = mano_output.center_joint.detach().cpu()
        joints_3d = joints_3d.view(N, T, 21, 3)
        # Apply ortographic projection to the 3D joints to obtain 2D image coordinates
        x = torch.matmul(R, joints_3d.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = x[...,:2]
        x *= S.unsqueeze(-1)
        x += O.unsqueeze(2)

        return x, pose_coeffs, betas, vertices, R, S, O, center_joint, joints_3d

    def training_step(self, batch):
        # Unpack the batch data
        _, x_or, x_masked, scores, masked_frames_idxs = batch
        # Forward pass through the model
        (logits, theta, beta, _, _, _, _, _, _) = self(x_masked)
        # Loss only applied on frames with masked joints
        valid_idxs = torch.where(masked_frames_idxs != -1.)
        logits = logits[valid_idxs]
        x_or = x_or[valid_idxs]
        scores = scores[valid_idxs]
        # Compute reconstruction loss (LRec) and regularization loss (LReg)
        lrec = torch.norm(logits[scores>self.eps] - x_or[scores>self.eps], p=1, dim=1).sum()
        beta_t_minus_one = torch.roll(beta, shifts=1, dims=1)
        beta_t_minus_one[:, 0] = 0.
        lreg = torch.norm(theta, 2) + self.weight_beta * torch.norm(beta, 2) + \
            self.weight_delta * torch.norm(beta - beta_t_minus_one, 2)
        # Combine both losses
        loss = lrec + (self.lmbd * lreg)
        # Append step loss 
        self.train_step_losses.append(loss.detach().cpu())
        if self.normalize_inputs: # If inputs to the network are normalized
            # Set means and stds attributes if they are not already
            if not hasattr(self, 'means') or not hasattr(self, 'stds'):
                self.means = torch.from_numpy(np.load(self.trainer.datamodule.MEANS_NPY_FPATH)).to(self.device)
                self.stds = torch.from_numpy(np.load(self.trainer.datamodule.STDS_NPY_FPATH)).to(self.device)
            # Reverse normalization
            logits = (logits * self.stds) + self.means
            x_or = (x_or * self.stds) + self.means
        # Compute PCK metrics
        self.train_pck_20(preds=logits, target=x_or)
        self.train_pck_auc_20_40(preds=logits, target=x_or)
        # Log metrics
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_PCK_20', self.train_pck_20, on_step=True, on_epoch=False)
        self.log('train_PCK_AUC_20-40', self.train_pck_auc_20_40, on_step=True, on_epoch=False)

        return loss

    def on_train_epoch_end(self):
        # Compute mean of step losses at the end of the epoch
        mean_epoch_loss = torch.stack(self.train_step_losses).mean()
        self.logger.experiment.add_scalars("losses", {"train_loss": mean_epoch_loss}, global_step=self.current_epoch)
        # Clear step losses placeholder
        self.train_step_losses.clear()

    def validation_step(self, batch, batch_idx):
        # Unpack batch data
        _, x_or, x_masked, scores, masked_frames_idxs = batch
        # Process data through the model
        (logits, beta, theta, _, _, _, _, _, _) = self(x_masked)
        # Loss is only applied on frames with masked joints
        valid_idxs = torch.where(masked_frames_idxs != -1.)
        logits = logits[valid_idxs]
        x_or = x_or[valid_idxs]
        scores = scores[valid_idxs]
        # Compute LRec and LReg
        lrec = torch.norm(logits[scores>self.eps] - x_or[scores>self.eps], p=1, dim=1).sum()
        beta_t_minus_one = torch.roll(beta, shifts=1, dims=1)
        beta_t_minus_one[:, 0] = 0.
        lreg = torch.norm(theta, 2) + self.weight_beta * torch.norm(beta, 2) + \
            self.weight_delta * torch.norm(beta - beta_t_minus_one, 2)
        # Combine both losses
        loss = lrec + (self.lmbd * lreg)
        # Append validation step loss
        self.val_step_losses.append(loss)
        if self.normalize_inputs: # If inputs to the network are normalized
            # Set means and stds attributes if they are not already
            if not hasattr(self, 'means') or not hasattr(self, 'stds'):
                self.means = torch.from_numpy(np.load(self.trainer.datamodule.MEANS_NPY_FPATH)).to(self.device)
                self.stds = torch.from_numpy(np.load(self.trainer.datamodule.STDS_NPY_FPATH)).to(self.device)
            # Reverse normalization
            logits = (logits * self.stds) + self.means
            x_or = (x_or * self.stds) + self.means
        # Compute PCK metrics
        self.val_pck_20(preds=logits, target=x_or)
        self.val_pck_auc_20_40(preds=logits, target=x_or)
        # Log metrics 
        self.log('val_loss', loss, on_step=False, prog_bar=True)
        self.log('val_PCK_20', self.val_pck_20, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_PCK_AUC_20_40', self.val_pck_auc_20_40, on_step=False, on_epoch=True)
        self.log("hp_metric", self.val_pck_20, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # Compute mean step losses at the end of the epoch
        mean_epoch_loss = torch.stack(self.val_step_losses).mean()
        self.logger.experiment.add_scalars("losses", {"val_loss": mean_epoch_loss}, global_step=self.current_epoch)
        # Clear step losses placeholder
        self.val_step_losses.clear()
        
    def configure_optimizers(self):
        toret = {}
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.use_onecycle_lr:
            lr_scheduler_config = dict(
                scheduler=torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=self.lr,
                    total_steps=self.total_steps,
                    pct_start=self.pct_start,
                    anneal_strategy='linear'
                )
            )
        toret['optimizer'] = optimizer
        if self.use_onecycle_lr:
            toret['lr_scheduler'] = lr_scheduler_config

        return toret