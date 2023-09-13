import torch
import lightning.pytorch as pl

from signbert.metrics.PCK import PCK, PCKAUC
from signbert.model.GestureExtractor import GestureExtractor
from signbert.model.hand_model.HandAwareModelDecoder import HandAwareModelDecoder
from IPython import embed; from sys import exit

class SignBertModel(pl.LightningModule):

    def __init__(self, in_channels, num_hid, num_heads, tformer_n_layers, eps, weight_beta, weight_delta, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.num_hid = num_hid
        self.num_heads = num_heads
        self.tformer_n_layers = tformer_n_layers
        self.eps = eps
        self.weight_beta = weight_beta
        self.weight_delta = weight_delta

        self.ge = GestureExtractor(in_channels=in_channels, num_hid=num_hid)
        el = torch.nn.TransformerEncoderLayer(d_model=num_hid*21, nhead=self.num_heads, batch_first=True)
        self.te = torch.nn.TransformerEncoder(el, num_layers=self.tformer_n_layers)
        self.hd = HandAwareModelDecoder(
            num_hid,
            n_pca_components=6, 
            mano_model_file='/home/gts/projects/jsoutelo/SignBERT+/signbert/model/hand_model/MANO_RIGHT_npy.pkl'
        )

        self.pck_20 = PCK(thr=20)
        self.pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)

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
        logits = logits[torch.where(masked_frames_idxs != -1.)]
        x_or = x_or[torch.where(masked_frames_idxs != -1.)]
        scores = scores[torch.where(masked_frames_idxs != -1.)]
        # Compute LRec
        lrec = torch.norm(logits[scores>self.eps] - x_or[scores>=self.eps], p=1, dim=1).sum()
        beta_t_minus_one = torch.roll(beta, shifts=1, dims=1)
        beta_t_minus_one[:, 0] = 0.
        lreg = torch.norm(theta, 2) + self.weight_beta * torch.norm(beta, 2) + \
            self.weight_delta * torch.norm(beta - beta_t_minus_one, 2)
        loss = lrec + lreg
        
        self.pck_20(preds=logits, target=x_or)
        self.pck_auc_20_40(preds=logits, target=x_or)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_PCK@20', self.pck_20, on_epoch=True)
        self.log('train_PCK_AUC@20-40', self.pck_auc_20_40, on_epoch=True)

        return loss
    
    # def on_train_epoch_end(self):
    #     self.log('train_PCK@20', self.pck_20.compute(), on_step=False)
    #     self.log('train_PCK_AUC@20_40', self.pck_auc_20_40.compute(), on_step=False)
    #     self.pck_20.reset()
    #     self.pck_auc_20_40.reset()

    def validation_step(self, batch, batch_idx):
        _, x_or, x_masked, _, masked_frames_idxs = batch
        (logits, theta, beta) = self(x_masked)
        logits = logits[torch.where(masked_frames_idxs != -1.)]
        x_or = x_or[torch.where(masked_frames_idxs != -1.)]
        
        self.pck_20(preds=logits, target=x_or)
        self.pck_auc_20_40(preds=logits, target=x_or)
        
        self.log('val_PCK@20', self.pck_20, on_step=False, on_epoch=True)
        self.log('val_PCK_AUC@20_40', self.pck_auc_20_40, on_step=False, on_epoch=True)

    # def on_validation_epoch_end(self):
    #     self.log('val_PCK@20', self.pck_20.compute(), on_step=False)
    #     self.log('val_PCK_AUC@20_40', self.pck_auc_20_40.compute(), on_step=False)

    #     self.pck_20.reset()
    #     self.pck_auc_20_40.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())

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

