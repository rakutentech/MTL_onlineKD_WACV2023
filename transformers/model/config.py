import yaml
from pathlib import Path

import os


def load_config():
    return yaml.load(
        open(Path(__file__).parent / "config.yml", "r"), Loader=yaml.FullLoader
    )


def check_os_environ(key, use):
    if key not in os.environ:
        raise ValueError(
            f"{key} is not defined in the os variables, it is required for {use}."
        )

class Args:
    MODEL = 'vit_tiny_patch16_384'
    device = 'cuda'
    
    num_threads = 2
    epoch = 0
    last_epoch = -1
    norm = 'linear'
    epochs = 200

    root = "."

    root_HPC = "/home/geethu.jacob/workspace_shared/MTL/RLW_Supplementary_Material/nyu_cityscapes/transformers/"
    epoch = 0

    # depth specific arguments
    same_lr = False
    wd = 0.1
    div_factor = 25
    final_div_factor = 100
    rank = 0
    n_bins = 256
    norm = "linear"
    dropout = 0.0
    drop_path = 0.1

    # model specific arguments
    backbone = MODEL
    decoder = 'mask_transformer'

    input_height = 384
    input_width = 384
    if MODEL=='vit_tiny_patch16_384':
        image_size = 384
        patch_size = 16
        d_model = 192
        n_heads = 12
        n_layers = 12
        normalization = 'vit'    
