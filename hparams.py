import os
import argparse
from functools import partial

import optuna
import lightning.pytorch as pl
from optuna.trial import TrialState, FrozenTrial
from optuna.study import MaxTrialsCallback
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from signbert.model.SignBertModelManoTorch import SignBertModel as SignBertModelManoTorch
from signbert.data_modules.HANDS17DataModule import HANDS17DataModule

from IPython import embed


TRAIN_EPOCHS = 1500
OPT_EPOCHS = 500
DIR = os.getcwd()

def objective(trial: optuna.trial.Trial, dpath, name, device, acc_grad_batches) -> float:
    # Data hparams
    R = 0.3 # fixed
    m = 5 # fixed
    K = 8 # fixed
    max_disturbance = 0.25 # fixed
    batch_size = 16 # fixed
    normalize = True # fixed

    # Model hparams
    in_channels = 2 # fixed
    num_hid = 32 # fixed
    num_heads = 4 # fixed
    tformer_n_layers = 1 # fixed
    tformer_dropout = 0.25 # fixed
    eps = 0.5 # fixed
    hand_cluster = False # fixed
    n_pca_components = trial.suggest_categorical('n_pca_components', [11, 15, 25, 35])
    gesture_extractor_cls = 'signbert.model.GestureExtractor.GestureExtractor' # fixed
    gesture_extractor_args = dict( # fixed
        in_channels=2,
        num_hid=32,
        dropout=0.25,
    )
    normalize_inputs = True # fixed
    use_pca = True # fixed
    flat_hand = False # fixed

    # Loss hparams
    lmbd = trial.suggest_float('lmdb', 1e-5, 1e-1)
    weight_beta = trial.suggest_float('weight_beta', 100, 800)
    weight_delta = trial.suggest_float('weight_delta', 100, 800)

    # Learning hparams
    use_onecycle_lr = True # fixed
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    pct_start = trial.suggest_float('pct_start', 0.05, 0.5, log=True)
    total_steps = TRAIN_EPOCHS # fixed
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    model_config=dict(
        in_channels=in_channels,
        num_hid=num_hid,
        num_heads=num_heads,
        tformer_n_layers=tformer_n_layers,
        tformer_dropout=tformer_dropout,
        eps=eps,
        lmbd=lmbd,
        weight_beta=weight_beta,
        weight_delta=weight_delta,
        lr=lr,
        hand_cluster=hand_cluster,
        n_pca_components=n_pca_components,
        gesture_extractor_cls=gesture_extractor_cls,
        gesture_extractor_args=gesture_extractor_args,
        total_steps=total_steps,
        normalize_inputs=normalize_inputs,
        use_pca=use_pca,
        flat_hand=flat_hand,
        weight_decay=weight_decay,
        use_onecycle_lr=use_onecycle_lr,
        pct_start=pct_start,
    )
    datamodule_config = dict(
        batch_size=batch_size,
        normalize=normalize,
        R=R,
        m=m,
        K=K,
        max_disturbance=max_disturbance,
    )
    
    seed_everything(42, workers=True)
    model = SignBertModelManoTorch(**model_config)
    datamodule = HANDS17DataModule(**datamodule_config)

    trainer = pl.Trainer(
        deterministic=True,
        logger=TensorBoardLogger(save_dir=dpath, name=name, default_hp_metric=False),
        default_root_dir=dpath,
        check_val_every_n_epoch=10,
        enable_checkpointing=False,
        max_epochs=OPT_EPOCHS,
        accelerator='gpu',
        strategy='auto',
        devices=[device],
        accumulate_grad_batches=acc_grad_batches, 
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            PyTorchLightningPruningCallback(trial, monitor="val_PCK_20"),
            # EarlyStopping(monitor="val_PCK_20", mode="max", patience=30, min_delta=1e-3), # 300 epochs patience (check_val_every_n_epoch * patience)
        ],
    )

    hyperparameters = dict(
        R=R,
        m=m,
        K=K,
        max_disturbance=max_disturbance,
        n_pca_components=n_pca_components,
        lmbd=lmbd,
        weight_beta=weight_beta,
        weight_delta=weight_delta,
        lr=lr,
        pct_start=pct_start,
        weight_decay=weight_decay,
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule)

    return trainer.callback_metrics["val_PCK_20"].item()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        '--name',
        type=str,
        default='test'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0
    )
    parser.add_argument(
        '--acc_grad_batches',
        type=int,
        default=1
    )
    args = parser.parse_args()

    logs_dpath = os.path.join(DIR, 'hparams_logs')

    storage_fpath = os.path.join(logs_dpath, args.name, f'{args.name}.db')
    if not os.path.isdir(os.path.split(storage_fpath)[0]):
        os.makedirs(os.path.split(storage_fpath)[0])
    # pruner = optuna.pruners.NopPruner()
    # pruner = optuna.pruners.MedianPruner(
    #     n_startup_trials=10,
    #     n_warmup_steps=20000,
    #     interval_steps=100,
    # )
    pruner = optuna.pruners.ThresholdPruner(
        lower=.5,
        n_warmup_steps=250,
        interval_steps=10
    )
    study = optuna.create_study(study_name=args.name, 
                                direction="maximize", pruner=pruner, 
                                storage=f'sqlite:///{storage_fpath}', 
                                load_if_exists=True)
    n_trials = None 
    study.optimize(
        partial(
            objective, 
            dpath=logs_dpath, 
            name=args.name, 
            device=args.device, 
            acc_grad_batches=args.acc_grad_batches
        ), 
        n_trials=n_trials, 
        timeout=None,
        # callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))],
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Define the file name
    file_name = os.path.join(logs_dpath, args.study_name, "result.txt")
    # Open the file in write mode
    with open(file_name, 'w') as file:
        file.write("Number of finished trials: {}\n".format(len(study.trials)))
        file.write("Best trial:\n")
        trial = study.best_trial
        file.write("  Value: {}\n".format(trial.value))
        file.write("  Params: \n")
        for key, value in trial.params.items():
            file.write("    {}: {}\n".format(key, value))