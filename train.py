import os
import argparse

from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from signbert.model.SignBertModel import SignBertModel
from signbert.data_modules.HANDS17DataModule import HANDS17DataModule

from IPython import embed; from sys import exit

_DEBUG = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()

    batch_size = 16
    epochs = 600
    lr = 1e-3
    hands17 = HANDS17DataModule(batch_size=batch_size)
    model = SignBertModel(
        in_channels=2, 
        num_hid=32, 
        num_heads=4, 
        tformer_n_layers=1, 
        eps=0.5, 
        lmbd=0.01, 
        weight_beta=10., 
        weight_delta=100.,
        total_steps=(hands17.NUM_SAMPLES // batch_size) * epochs,
        lr=lr
    )
    
    if _DEBUG:
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
            devices=1,
            max_epochs=epochs
        )

    if args.ckpt:
        print('Resuming training from ckpt:', args.ckpt)
    
    lr_logger = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    ckpt_dirpath = os.path.join(tb_logger.log_dir, 'ckpts')
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dirpath, save_top_k=10, monitor="val_PCK@20", mode='max', filename="epoch={epoch:02d}-step={step}-{val_PCK@20:.4f}", save_last=True)
    
    trainer = Trainer(**trainer_config, callbacks=[lr_logger, checkpoint_callback])
    trainer.fit(model, hands17, ckpt_path=args.ckpt)