from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.tasks import yaml_model_load, initialize_weights, torch_safe_load, guess_model_task

from ultralytics.yolo.utils.torch_utils import intersect_dicts, make_divisible #smart_inference_mode, de_parallel,
from ultralytics.yolo.utils import LOGGER

from loss import v8DetectionPointLoss


class DetectionPointModel(DetectionModel):
    """Detection model using point loss"""
    def load(self, weights, verbose=True):
        """Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        print(f'Loading weights [{type(weights)}] to {type(self)}...')
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

    def init_criterion(self):
        return v8DetectionPointLoss(self)
    