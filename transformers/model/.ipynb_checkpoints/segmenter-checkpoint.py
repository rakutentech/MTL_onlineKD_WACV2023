import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import padding, unpadding
from timm.models.layers import trunc_normal_


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder_seg,
        decoder_depth,
        decoder_sn,
        args
    ):
        super().__init__()
        
        self.patch_size = encoder.patch_size
        self.predictions = {}
        self.encoder = encoder
        self.features = encoder.features
        self.stages = encoder.stages
        self.weights = encoder.weights 
        self.decoder_seg = decoder_seg
        self.decoder_depth = decoder_depth
        self.decoder_sn = decoder_sn
        self.args = args

    
    def segmentation_head(self, x,H,W):
        decoder = self.decoder_seg
        masks = decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H, W))
        return masks
    
    def depth_head(self,x,H,W):
        decoder = self.decoder_depth  
        
        masks_all = decoder(x, (H, W))
        
        masks = F.interpolate(masks_all, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H, W))
            
        return masks_all, masks
    
    def sn_head(self,x, H,W):
        decoder = self.decoder_sn    
        masks_all = decoder(x, (H, W))
        masks = F.interpolate(masks_all, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H, W))
            
        return masks
    
