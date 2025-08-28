"""
A helper function to get a default model for quick testing
"""
import os
from omegaconf import open_dict
from hydra import compose, initialize

import torch
from LCCSF.model.LCCSF import LCCSF
from LCCSF.inference.utils.args_utils import get_dataset_cfg
from LCCSF.utils.download_models import download_models_if_needed


def get_default_model() -> LCCSF:
    initialize(version_base='1.3.2', config_path="../config", job_name="eval_config")
    cfg = compose(config_name="eval_config")

    weight_dir = download_models_if_needed()
    with open_dict(cfg):
        cfg['weights'] = os.path.join(weight_dir, 'DAVIS.pth')
    get_dataset_cfg(cfg)

    # Load the network weights
    LCCSF = LCCSF(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights)
    LCCSF.load_weights(model_weights)

    return LCCSF
