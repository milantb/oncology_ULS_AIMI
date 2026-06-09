import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable, Optional


from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.focalloss2 import FocalLossFunction
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerFocalLoss1000(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000

    def _build_loss(self):
        self.print_to_log_file("NN-UNet with focal loss: (1000) epochs)")
        loss = FocalLossFunction(
            apply_nonlin=softmax_helper_dim1,
            gamma=2.0,
            alpha=[0.25, 0.75],
            do_bg=True,
            smooth=1e-5,
        )


        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
        
