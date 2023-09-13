import argparse
from lightning.pytorch import Trainer, seed_everything

from signbert.model.SignBertModel import SignBertModel
from signbert.data_modules.HANDS17DataModule import HANDS17DataModule


_DEBUG = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()

    batch_size = 16
    hands17 = HANDS17DataModule(batch_size=batch_size)
    model = SignBertModel(in_channels=2, num_hid=32, num_heads=4, tformer_n_layers=1, eps=0.3, weight_beta=0.2, weight_delta=0.2)
    
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
            max_epochs=15000
        )

    if args.ckpt:
        print('Resuming training from ckpt:', args.ckpt)

    trainer = Trainer(**trainer_config)
    trainer.fit(model, hands17, ckpt_path=args.ckpt)