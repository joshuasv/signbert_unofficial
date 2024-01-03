import os
import shutil
import argparse
from pprint import pprint

import yaml
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from signbert.model.SignBertModelManoTorch import SignBertModel as SignBertModelManoTorch
from signbert.model.PretrainSignBertModelManoTorch import SignBertModel as PretrainSignBert
from signbert.data_modules.HANDS17DataModule import HANDS17DataModule
from signbert.data_modules.PretrainDataModule import PretrainDataModule

from IPython import embed; from sys import exit

_DEBUG = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--val-interval', default=None, type=int)
    args = parser.parse_args()
    # Load config 
    with open(args.config, 'r') as fid:
        cfg = yaml.load(fid, yaml.SafeLoader)
    pprint(cfg)
    epochs = args.epochs if args.epochs is not None else 600 
    lr = args.lr if args.lr is not None else cfg['lr'] # Preference over arguments
    batch_size = cfg['batch_size']
    normalize = cfg['normalize']
    pretrain = cfg.get("pretrain", False)
    datasets = cfg.get("datasets", None)
    if pretrain: # If pretraining is to be executed
        assert datasets is not None
        # Initialize datamodule
        datamodule = PretrainDataModule(
            datasets,
            batch_size=batch_size,
            normalize=normalize
        )
        # Initialize model
        model = PretrainSignBert(
            **cfg["model_args"],
            lr=lr, 
            normalize_inputs=normalize, 
        )
    else:
        # Initialize datamodule
        datamodule = HANDS17DataModule(
            batch_size=batch_size, 
            normalize=normalize, 
            **cfg.get('dataset_args', dict())
        )
        # Initialize model
        mano_model_cls = SignBertModelManoTorch
        model = mano_model_cls(
            **cfg['model_args'], 
            lr=lr, 
            normalize_inputs=normalize, 
            means_fpath=HANDS17DataModule.MEANS_NPY_FPATH, 
            stds_fpath=HANDS17DataModule.STDS_NPY_FPATH,
        )
    
    if _DEBUG: # Switch between trainer configs wheter debug is enabled
        trainer_config = dict(
            accelerator='cpu',
            strategy='auto',
            devices=1,
        )
        n_parameters = 0
        for p in model.parameters():
            if p.requires_grad:
                n_parameters += p.numel()
        print('# params:', n_parameters)
    else:
        trainer_config = dict(
            accelerator='gpu',
            strategy='auto',
            devices=[args.device],
            max_epochs=epochs
        )

    if args.ckpt: # Check if training shall be resumed from checkpoint
        print('Resuming training from ckpt:', args.ckpt)
    # Initialize learning rate logger
    lr_logger = LearningRateMonitor(logging_interval='step')
    # Initialize Tensorboard logger 
    logs_dpath = os.path.join(os.getcwd(), 'logs')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_dpath, name=args.name)
    # Initialize model checkpointing callback
    ckpt_dirpath = os.path.join(tb_logger.log_dir, 'ckpts')
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dirpath, save_top_k=10, monitor="val_PCK_20", mode='max', filename="epoch={epoch:02d}-step={step}-{val_PCK_20:.4f}", save_last=True)
    # Initialize early stopping callback
    early_stopping_callback = EarlyStopping(monitor="val_PCK_20", mode="max", patience=30, min_delta=1e-4)
    # Setup and configure the Trainer 
    trainer = Trainer(
        **trainer_config,
        accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1), 
        logger=tb_logger, 
        callbacks=[lr_logger, checkpoint_callback],#, early_stopping_callback],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=cfg.get('precision', '32-true'),
        check_val_every_n_epoch=args.val_interval
    )
    trainer.fit(model, datamodule, ckpt_path=args.ckpt) # Start trainig