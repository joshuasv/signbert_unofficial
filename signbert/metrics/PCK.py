import torch
import numpy as np
from torch import Tensor
from torch.nn import ModuleList
from torchmetrics import Metric, MetricCollection

from IPython import embed; from sys import exit


class PCK(Metric):
    """Percentage of Correct Keypoints metric."""

    def __init__(self, thr: float = 20.):
        super().__init__()
        self.threshold = thr
        # self.add_state('threshold', torch.tensor(thr), dist_reduce_fx=None)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        assert preds.shape == target.shape
        distances = torch.norm(target - preds, dim=-1)
        correct = (distances < self.threshold).sum()
        self.correct += correct
        self.total += distances.numel()

    def compute(self): return self.correct.float() / self.total


class PCKAUC(Metric):

    def __init__(self, thr_min: float = 20, thr_max: float = 40):
        super().__init__()
        assert thr_min < thr_max
        step = 1
        thresholds = torch.arange(thr_min, thr_max+step, step)
        
        self.metrics = ModuleList([PCK(thr) for thr in thresholds])
        self.add_state("thresholds", default=thresholds, dist_reduce_fx=None)
        self.add_state("diff", default=torch.tensor(thr_max-thr_min), dist_reduce_fx=None)
        # self.add_state("auc", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor):
        assert preds.shape == target.shape
        
        for m in self.metrics: m.update(preds, target)

    def compute(self):
        result = torch.cat([m.compute().reshape(1) for m in self.metrics])
        # Normalize so its between [0,1]
        return torch.trapz(result, self.thresholds) / self.diff
    
    def reset(self):
        self._update_count = 0
        self._forward_cache = None
        self._computed = None
        # reset internal states
        self._cache = None
        self._is_synced = False

        for m in self.metrics: m.reset()
        self.auc = 0.


if __name__ == '__main__':

    gt = torch.rand(16, 2361, 21, 2)
    pred = torch.rand(16, 2361, 21, 2)

    pck = PCK()
    pck.update(gt, pred)
    print(f'{pck.compute()=}')

    pck_auc = PCKAUC()
    pck_auc.update(gt, pred)
    print(f'{pck_auc.compute()=}')
    embed(); exit()