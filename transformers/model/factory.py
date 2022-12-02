from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from model.vit import VisionTransformer
from model.utils import checkpoint_filter_fn
from model.decoder import DecoderSN, Decoderdepth
from model.decoder import MaskTransformer
from model.segmenter import Segmenter


def create_vit(model_cfg,args):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    model = VisionTransformer(args,**model_cfg)
    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
        print('Model loaded successfully')
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model



def create_decoder_sn(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    n_bins= decoder_cfg.pop("n_bins")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size
    
    dim = encoder.d_model
    n_heads = dim // 64
    decoder_cfg["n_heads"] = n_heads
    decoder_cfg["d_model"] = dim
    decoder_cfg["d_ff"] = 4 * dim
    decoder = DecoderSN(**decoder_cfg)
    return decoder

def create_decoder_segmenter(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size


    dim = encoder.d_model
    n_heads = dim // 64
    decoder_cfg["n_heads"] = n_heads
    decoder_cfg["d_model"] = dim
    decoder_cfg["d_ff"] = 4 * dim
    decoder = MaskTransformer(**decoder_cfg)
    return decoder

def create_decoder_depth(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    n_bins= decoder_cfg.pop("n_bins")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size
    decoder_cfg["n_cls"]=1
    dim = encoder.d_model
    n_heads = dim // 64
    decoder_cfg["n_heads"] = n_heads
    decoder_cfg["d_model"] = dim
    decoder_cfg["d_ff"] = 4 * dim
        
    decoder = Decoderdepth(**decoder_cfg)
    
    return decoder




def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant.net_kwargs

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant

def create_model(model_cfg,args):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    
    
    encoder = create_vit(model_cfg,args)
    # create a single function for create decoder
    decoder_seg=[];decoder_depth=[]; decoder_sn=[]
    if 'segmentation' in args.tasks:
        decoder_cfg["n_cls"] = model_cfg["n_cls"]
        decoder_seg = create_decoder_segmenter(encoder, decoder_cfg)
    if 'depth' in args.tasks:
        decoder_cfg["n_bins"]=args.n_bins
        decoder_depth = create_decoder_depth(encoder, decoder_cfg)
    if 'normal' in args.tasks:
        decoder_cfg["n_cls"] = 180
        decoder_cfg["n_bins"]=args.n_bins
        decoder_sn = create_decoder_sn(encoder, decoder_cfg)
        
    model = Segmenter( encoder, decoder_seg,decoder_depth, decoder_sn,args)
    
    return model

