"""Metrics for measuring the topology of binary segmentation masks."""

import numpy as np
import torch
from skimage.morphology import skeletonize

from torchtrainer.util.post_processing import logits_to_preds

type CpuOrCudaTensor = torch.Tensor

class ClDice:
    """Calculate the cldice metric for a batch of data."""
    
    """
        Imported from: torchtrainer
        https://github.com/chcomin/torchtrainer/ 
    """
    def __init__(self, reduction: str = "mean"):
        """
        Parameters
        ----------
        reduction
            Reduction method. Can be 'mean' or 'none'.
        """
        self.reduction = reduction

    def __call__(
            self, 
            scores: CpuOrCudaTensor, 
            targets: CpuOrCudaTensor,
            logits_flag: bool = False
            ) -> np.ndarray | float:
        """
        Parameters
        ----------
        scores
            Output from a network. Dimension 1 is treated as the class dimension.
        targets
            Labels
        logits_flag
            If this flag is true, the function will convert the inputed logits to predictions. 
            If the flag is false, the function will assume that the input is the predictions and no convertion will be made.
        """
        return cl_dice(scores, targets, self.reduction, logits_flag)

@torch.no_grad()
def cl_dice(
    scores: CpuOrCudaTensor, 
    targets: CpuOrCudaTensor, 
    reduction: str = "mean",
    logits_flag: bool = False
    ) -> torch.Tensor:
    """Calculate the clDice metric for a batch of data."""

    if logits_flag:
        preds = logits_to_preds(scores, return_indices=True, keepdim=False)
    else:
        preds=scores

    preds = preds.cpu().numpy()>0
    targets = targets.cpu().numpy()>0

    scores = []
    for pred, target in zip(preds, targets):
        pred_skel = skeletonize(pred)
        target_skel = skeletonize(target)

        tprec = (pred_skel & target).sum()/pred_skel.sum()
        tsens = (target_skel & pred).sum()/target_skel.sum()

        score = 2*tprec*tsens/(tprec+tsens)
        scores.append(score)

    scores = torch.tensor(scores)

    if reduction == "mean":
        scores = scores.mean()
    elif reduction == "none":
        pass

    return scores