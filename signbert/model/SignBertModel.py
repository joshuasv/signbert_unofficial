import torch
import lightning.pytorch as pl

from signbert.metrics.PCK import PCK, PCKAUC
from signbert.model.GestureExtractor import GestureExtractor
from signbert.model.hand_model.HandAwareModelDecoder import HandAwareModelDecoder
from IPython import embed; from sys import exit

class SignBertModel(pl.LightningModule):

    def __init__(
            self, 
            in_channels, 
            num_hid, 
            num_heads, 
            tformer_n_layers, 
            eps, 
            lmbd, 
            weight_beta, 
            weight_delta,
            total_steps,
            lr,
            *args, 
            **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.num_hid = num_hid
        self.num_heads = num_heads
        self.tformer_n_layers = tformer_n_layers
        self.eps = eps
        self.lmbd = lmbd
        self.weight_beta = weight_beta
        self.weight_delta = weight_delta
        self.total_steps = total_steps
        self.lr = lr

        self.ge = GestureExtractor(in_channels=in_channels, num_hid=num_hid)
        el = torch.nn.TransformerEncoderLayer(d_model=num_hid*21, nhead=self.num_heads, batch_first=True)
        self.te = torch.nn.TransformerEncoder(el, num_layers=self.tformer_n_layers)
        self.hd = HandAwareModelDecoder(
            num_hid,
            n_pca_components=6, 
            mano_model_file='/home/gts/projects/jsoutelo/SignBERT+/signbert/model/hand_model/MANO_RIGHT_npy.pkl'
        )

        self.train_pck_20 = PCK(thr=20)
        self.train_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        self.val_pck_20 = PCK(thr=20)
        self.val_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)

    def forward(self, x):
        x = self.ge(x)
        # Remove last dimension M and permute to be (N, T, C, V)
        x = x.squeeze(-1).permute(0, 2, 1, 3)
        N, T, C, V = x.shape
        x = x.reshape(N, T, C*V)
        x = self.te(x)
        x, theta, beta = self.hd(x)

        return x, theta, beta

    def training_step(self, batch):
        _, x_or, x_masked, scores, masked_frames_idxs = batch
        (logits, theta, beta) = self(x_masked)

        # Loss only applied on frames with masked joints
        valid_idxs = torch.where(masked_frames_idxs != -1.)
        logits = logits[valid_idxs]
        x_or = x_or[valid_idxs]
        scores = scores[valid_idxs]
        # Compute LRec
        lrec = torch.norm(logits[scores>self.eps] - x_or[scores>=self.eps], p=1, dim=1).sum()
        beta_t_minus_one = torch.roll(beta, shifts=1, dims=1)
        beta_t_minus_one[:, 0] = 0.
        lreg = torch.norm(theta, 2) + self.weight_beta * torch.norm(beta, 2) + \
            self.weight_delta * torch.norm(beta - beta_t_minus_one, 2)
        loss = lrec + (self.lmbd * lreg)
        
        self.train_pck_20(preds=logits, target=x_or)
        self.train_pck_auc_20_40(preds=logits, target=x_or)

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_PCK@20', self.train_pck_20, on_step=True, on_epoch=False)
        self.log('train_PCK_AUC@20-40', self.train_pck_auc_20_40, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        _, x_or, x_masked, _, masked_frames_idxs = batch
        (logits, theta, beta) = self(x_masked)
        logits = logits[torch.where(masked_frames_idxs != -1.)]
        x_or = x_or[torch.where(masked_frames_idxs != -1.)]
        
        self.val_pck_20(preds=logits, target=x_or)
        self.val_pck_auc_20_40(preds=logits, target=x_or)
        
        self.log('val_PCK@20', self.val_pck_20, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_PCK_AUC@20_40', self.val_pck_auc_20_40, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        # lr_scheduler_config = dict(
        #     scheduler=torch.optim.lr_scheduler.OneCycleLR(
        #         optimizer, 
        #         max_lr=1e-4,
        #         total_steps=self.total_steps,
        #         pct_start=0.1,
        #         anneal_strategy='linear'
        #     )
        # )

        return dict(
            optimizer=optimizer,
            # lr_scheduler=lr_scheduler_config
        )

        return optimizer

if __name__ == '__main__':
    import numpy as np
    from signbert.data_modules.MaskKeypointDataset import MaskKeypointDataset

    dataset = MaskKeypointDataset(
        npy_fpath='/home/temporal2/jsoutelo/datasets/HANDS17/preprocess/X_train.npy',
        R=0.2,
        m=5,
        K=6
    )
    # TODO
    # get parameters from paper
    model = SignBertModel(in_channels=2, num_hid=256, num_heads=4, tformer_n_layers=2, eps=0.3, weight_beta=0.2, weight_delta=0.2)

    seq, score, masked_frames_idxs = dataset[0]
    seq = torch.tensor(np.stack((seq, seq)).astype(np.float32))
    masked_frames_idxs = torch.tensor(np.stack((masked_frames_idxs,masked_frames_idxs)).astype(np.int32))
    batch = (seq, masked_frames_idxs)

    out = model(batch[0])

    print(f'{out[0].shape=}')
    embed(); exit()

