import torch
import numpy as np
from torch import Tensor
from torch.nn import ModuleList
from torchmetrics import Metric, MetricCollection

from IPython import embed; from sys import exit


class PCK(Metric):
    """
    Percentage of Correct Keypoints (PCK) metric class.

    This class extends PyTorch's Metric class to calculate the PCK metric. 
    PCK measures the percentage of predicted keypoints that are within a certain 
    threshold distance from the ground truth keypoints.

    Attributes:
    threshold (float): The distance threshold within which a keypoint is considered correctly predicted.
    correct (Tensor): The count of keypoints correctly predicted within the threshold.
    total (Tensor): The total count of keypoints predicted.
    """
    def __init__(self, thr: float = 20.):
        """
        Initialize the PCK metric.

        Parameters:
        thr (float): The threshold for considering a keypoint as correctly predicted. Default is 20.
        """
        super().__init__()
        self.threshold = thr
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        Update the state of the metric with new predictions and targets.

        Parameters:
        preds (Tensor): Predicted keypoints.
        target (Tensor): Ground truth keypoints.
        """
        assert preds.shape == target.shape
        # Calculate the L2 distance between predictions and targets
        distances = torch.norm(target - preds, dim=-1)
        # Count how many predictions are within the threshold distance
        correct = (distances < self.threshold).sum()
        self.correct += correct
        self.total += distances.numel()

    def compute(self): 
        """
        Compute the PCK value.

        Returns:
        float: The percentage of correctly predicted keypoints.
        """
        return self.correct.float() / self.total


class PCKAUC(Metric):
    """
    Area Under Curve (AUC) for the Percentage of Correct Keypoints (PCK) metric class.

    This class calculates the AUC of the PCK metric across a range of thresholds. 

    Attributes:
    metrics (ModuleList): A list of PCK metrics, each with a different threshold.
    thresholds (Tensor): A tensor of thresholds used to calculate PCK.
    diff (Tensor): The difference between the maximum and minimum thresholds.
    """
    def __init__(self, thr_min: float = 20, thr_max: float = 40):
        """
        Initialize the PCKAUC metric.

        Parameters:
        thr_min (float): The minimum threshold for PCK calculation. Default is 20.
        thr_max (float): The maximum threshold for PCK calculation. Default is 40.
        """
        super().__init__()
        assert thr_min < thr_max
        step = 1
        thresholds = torch.arange(thr_min, thr_max+step, step)
        # Create a PCK metric for each threshold
        self.metrics = ModuleList([PCK(thr) for thr in thresholds])
        self.add_state("thresholds", default=thresholds, dist_reduce_fx=None)
        self.add_state("diff", default=torch.tensor(thr_max-thr_min), dist_reduce_fx=None)
    
    def update(self, preds: Tensor, target: Tensor):
        """
        Update the state of the metric with new predictions and targets.

        Parameters:
        preds (Tensor): Predicted keypoints.
        target (Tensor): Ground truth keypoints.
        """
        assert preds.shape == target.shape
        # Update each PCK metric with the new predictions and targets
        for m in self.metrics: m.update(preds, target)

    def compute(self):
        """
        Compute the PCKAUC value.

        Returns:
        Tensor: The AUC of the PCK metric across the range of thresholds.
        """
        # Calculate PCK for each threshold and concatenate results
        result = torch.cat([m.compute().reshape(1) for m in self.metrics])
        # Calculate the area under the curve (AUC) and normalize so its between [0,1]
        return torch.trapz(result, self.thresholds) / self.diff
    
    def reset(self):
        """Reset the state of the metric."""
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