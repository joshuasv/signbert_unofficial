# python finetune.py --config finetune/configs/ISLR_MSASL.yml --ckpt logs/pretrain/version_0/ckpts/last.ckpt
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics import Accuracy

from finetune.ISLR.Head import Head
from signbert.model.PretrainSignBertModelManoTorch import SignBertModel as BaseModel
from IPython import embed


class SignBertModel(pl.LightningModule):

    def __init__(self, ckpt, lr, head_args):
        super().__init__()
        self.lr = lr

        self.model = BaseModel.load_from_checkpoint(ckpt, map_location="cpu")
        self._init_base_model()
        
        ge_hid_dim = self.model.hparams.gesture_extractor_args["hid_dim"]
        in_channels = ge_hid_dim[-1] if isinstance(ge_hid_dim, list) else ge_hid_dim
        self.head = Head(in_channels=in_channels, **head_args)

        num_classes = head_args["num_classes"]
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def _init_base_model(self):
        del self.model.pg
        del self.model.lhand_hd
        del self.model.rhand_hd
        del self.model.train_pck_20
        del self.model.train_pck_auc_20_40
        del self.model.val_pck_20
        del self.model.val_pck_auc_20_40
        self.model.freeze()

    def forward(self, arms, rhand, lhand):
        x = torch.concat((rhand, lhand), dim=2)
        # Extract hand tokens
        rhand, lhand = self.model.ge(x)
        rhand = rhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        lhand = lhand.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        # Extract arm tokens
        rarm, larm = self.model.stpe(arms)
        rarm = rarm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        larm = larm.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        N, T, C, V = rhand.shape
        # Add positional tokens
        rhand = rhand + rarm 
        lhand = lhand + larm 
        rhand = rhand.view(N, T, C*V)
        lhand = lhand.view(N, T, C*V)
        rhand = self.model.pe(rhand) 
        lhand = self.model.pe(lhand) 
        # Transformer
        rhand = self.model.te(rhand)
        lhand = self.model.te(lhand)
        # Head
        x = self.head(rhand, lhand)

        return x

    def training_step(self, batch):
        arms = batch["arms"]
        lhand = batch["lhand"]
        rhand = batch["rhand"]
        labels = batch["class_id"]
        logits = self(arms, rhand, lhand)
        loss = F.cross_entropy(logits, labels)
        acc = self.train_acc(logits, labels) 
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=False)

        return loss

    def validation_step(self, batch, dataloader_idx=0):
        arms = batch["arms"]
        lhand = batch["lhand"]
        rhand = batch["rhand"]
        labels = batch["class_id"]
        logits = self(arms, rhand, lhand)
        loss = F.cross_entropy(logits, labels)
        acc = self.val_acc(logits, labels) 
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        self.log("val_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer


if __name__ == "__main__":
    from finetune.ISLR.MSASLDataModule import MSASLDataModule
    head_args = {
        "num_classes": 32
    }
    model = SignBertModel(ckpt="logs/pretrain/version_0/ckpts/last.ckpt", lr=1e-3, head_args=head_args)
    d = MSASLDataModule(batch_size=32, normalize=True)
    d.setup(stage="fit")
    dl = d.train_dataloader()
    batch = next(iter(dl))
    out = model(arms=batch["arms"], lhand=batch["lhand"], rhand=batch["rhand"])
    embed(); exit()