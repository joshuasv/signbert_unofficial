import os
import argparse
from pprint import pformat

import yaml
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from finetune.ISLR.MSASLDataModule import MSASLDataModule
from finetune.SignBERTModel import SignBertModel

from IPython import embed;

class Config:

    def __init__(self, **config):
        self.__dict__.update(config)
    
    def __repr__(self):
        return pformat(vars(self), indent=2)


def main(args):
    with open(args.config, "r") as fid:
        cfg = yaml.load(fid, yaml.SafeLoader)
    # args overrides cfg
    cfg.update(args.__dict__)
    config = Config(**cfg)
    print(config)

    datamodule = MSASLDataModule(
        batch_size=config.batch_size,
        **config.datamodule_args
    )
    model = SignBertModel(ckpt=config.ckpt, lr=config.lr, head_args=config.head_args)

    logs_dpath = os.path.join(os.getcwd(), "finetune_logs")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_dpath, name=config.name)
    ckpt_dirpath = os.path.join(tb_logger.log_dir, "ckpts")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dirpath, 
        save_top_k=5, 
        monitor="val_acc", 
        mode="max", 
        filename="epoch={epoch:02d}-step={step}-{val_acc:.4f}", 
        save_last=True
    )
    trainer = Trainer(
        accelerator="gpu",
        strategy="auto",
        devices=[config.device],
        max_epochs=config.epochs,
        logger=tb_logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=config.precision
    )
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--name", default="test", type=str)
    parser.add_argument("--precision", default="32-true", type=str)
    args = parser.parse_args()

    main(args)