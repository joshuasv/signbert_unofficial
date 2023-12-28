import torch
import numpy as np
import lightning.pytorch as pl

from signbert.utils import my_import
from signbert.model.PositionalEncoding import PositionalEncoding
from signbert.metrics.PCK import PCK, PCKAUC
from manotorch.manolayer import ManoLayer, MANOOutput
from IPython import embed; from sys import exit


class SignBertModel(pl.LightningModule):

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

        num_hid_mult = 1 if hand_cluster else 21

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
        self.rhand_hd = ManoLayer(
            center_idx=0,
            flat_hand_mean=flat_hand,
            mano_assets_root='/home/gts/projects/jsoutelo/SignBERT+/thirdparty/manotorch/assets/mano',
            use_pca=use_pca,
            ncomps=n_pca_components,
        )
        self.lhand_hd = ManoLayer(
            center_idx=0,
            flat_hand_mean=flat_hand,
            mano_assets_root='/home/gts/projects/jsoutelo/SignBERT+/thirdparty/manotorch/assets/mano',
            use_pca=use_pca,
            ncomps=n_pca_components,
            side="left"
        )
        self.train_pck_20 = PCK(thr=20)
        self.train_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        self.val_pck_20 = PCK(thr=20)
        self.val_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)

        self.mean_loss = []
        self.mean_pck_20 = []

    def forward(self, arms, rhand, lhand):
        x = torch.concat((rhand, lhand), dim=2)
        # Extract hand tokens
        rhand, lhand = self.ge(x)
        rhand = rhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        lhand = lhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        # Extract arm tokens
        rarm, larm = self.stpe(arms)
        rarm = rarm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        larm = larm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        N, T, C, V = rhand.shape
        # Add positional tokens
        rhand = rhand + rarm 
        lhand = lhand + larm 
        rhand = rhand.view(N, T, C*V)
        lhand = lhand.view(N, T, C*V)
        rhand = self.pe(rhand) 
        lhand = self.pe(lhand) 
        # Transformer
        rhand = self.te(rhand)
        lhand = self.te(lhand)
        # Camera parameters
        rhand_params = self.pg(rhand)
        lhand_params = self.pg(lhand)
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
        # MANO
        rhand_mano_output: MANOOutput = self.rhand_hd(rhand_pose_coeffs, rhand_betas)
        lhand_mano_output: MANOOutput = self.lhand_hd(lhand_pose_coeffs, lhand_betas)
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
        # Ortographic projection
        rhand = torch.matmul(rhand_R, rhand_joints_3d.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        rhand = rhand[...,:2]
        rhand *= rhand_S.unsqueeze(-1)
        rhand += rhand_O.unsqueeze(2)
        lhand = torch.matmul(lhand_R, lhand_joints_3d.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        lhand = lhand[...,:2]
        lhand *= lhand_S.unsqueeze(-1)
        lhand += lhand_O.unsqueeze(2)

        return {
            "rhand": (rhand, rhand_pose_coeffs, rhand_betas, rhand_vertices, rhand_R, rhand_S, rhand_O, rhand_center_joint, rhand_joints_3d),
            "lhand": (lhand, lhand_pose_coeffs, lhand_betas, lhand_vertices, lhand_R, lhand_S, lhand_O, lhand_center_joint, lhand_joints_3d)
        }

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        for k, v in batch.items():
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
            hand_data = self(arms, rhand_masked, lhand_masked)
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
            # Compute LRec
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
            # Manual backward pass
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            if isinstance(sch, torch.optim.lr_scheduler.OneCycleLR):
                sch.step()
            # Compute metrics
            if self.normalize_inputs:
                # Check that means and stds are in the same device as trainer
                if self.device != self.trainer.datamodule.means[k].device:
                   self.trainer.datamodule.means[k] = self.trainer.datamodule.means[k].to(self.device)
                if self.device != self.trainer.datamodule.stds[k].device:
                    self.trainer.datamodule.stds[k] = self.trainer.datamodule.stds[k].to(self.device)
                means = self.trainer.datamodule.means[k]
                stds = self.trainer.datamodule.stds[k]
                rhand_logits = (rhand_logits * stds) + means
                lhand_logits = (lhand_logits * stds) + means
                rhand = (rhand * stds) + means
                lhand = (lhand * stds) + means
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
        dataset_key = list(self.trainer.datamodule.val_dataloaders.keys())[dataloader_idx]
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
        hand_data = self(arms, rhand_masked, lhand_masked)
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
        # Compute LRec
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
            means = self.trainer.datamodule.means[dataset_key]
            stds = self.trainer.datamodule.stds[dataset_key]
            rhand_logits = (rhand_logits * stds) + means
            lhand_logits = (lhand_logits * stds) + means
            rhand = (rhand * stds) + means
            lhand = (lhand * stds) + means
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