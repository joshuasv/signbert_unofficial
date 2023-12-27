import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.utilities import CombinedLoader

from signbert.utils import my_import


class PretrainDataModule(pl.LightningDataModule):

    def __init__(self, datasets, batch_size, normalize=False, mode="sequential"):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.normalize = normalize
        self.mode = mode
        self.means = {}
        self.stds = {}

    def prepare_data(self):
        for v in self.datasets.values():
            module_cls = my_import(v["module_cls"])
            dataset_args = v.get("dataset_args", dict())
            data_module = module_cls(
                batch_size=self.batch_size, 
                normalize=self.normalize, 
                **dataset_args
            )
            data_module.prepare_data()
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataloaders = {}
            self.val_dataloaders = {}
            for k, v in self.datasets.items():
                module_cls = my_import(v["module_cls"])
                dataset_args = v.get("dataset_args", dict())
                data_module = module_cls(
                    batch_size=self.batch_size, 
                    normalize=self.normalize, 
                    **dataset_args
                )
                data_module.setup()
                self.train_dataloaders[k] = data_module.train_dataloader()
                self.val_dataloaders[k] = data_module.val_dataloader()
                self.means[k] = torch.from_numpy(np.load(data_module.means_fpath))
                self.stds[k] = torch.from_numpy(np.load(data_module.stds_fpath))
    
    def train_dataloader(self):
        return CombinedLoader(self.train_dataloaders)
    
    def val_dataloader(self):
        return CombinedLoader(self.val_dataloaders, mode="sequential")