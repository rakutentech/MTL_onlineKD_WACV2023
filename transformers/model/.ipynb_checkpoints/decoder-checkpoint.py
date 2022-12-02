import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_

from model.blocks import Block, FeedForward
from model.utils import init_weights


            
class DecoderSN(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        n_cls=3
        n_layers=2
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, d_model, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.dot_product_layer = PixelWiseDotProduct()
        self.proj_patch = nn.Parameter(torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(torch.randn(d_model, d_model))
        self.decoder_norm = nn.LayerNorm(d_model)

        self.head1 = nn.Sequential(nn.Linear(self.d_model, 128),
                                   nn.Tanh(),
                                   nn.Linear(128, 64),
                                   nn.Tanh(),
                                   nn.Linear(64, n_cls)
                                  )
        
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)
        trunc_normal_(self.proj_patch, std=0.02)
        trunc_normal_(self.proj_classes, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        size_x = x.shape[1]
        masks_all=[]

        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        patches, cls_seg_feat = x[:, : -self.d_model], x[:, -self.d_model :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.head1(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
               
        return masks
    
    


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        
        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

            


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K,h,w):
        n, c, emb = x.size()
        _, cout, ck = K.size()
        assert emb == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x, K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)

    
    
class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=16, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        encoder_layers = nn.TransformerEncoderLayer(self.embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)  # takes shape S,N,E
    def forward(self, embeddings):        
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        
        return x
    


    

    
class Decoderdepth(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        n_cls=3
        n_layers=2
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, d_model, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.dot_product_layer = PixelWiseDotProduct()
        self.proj_patch = nn.Parameter(torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(torch.randn(d_model, d_model))
        self.decoder_norm = nn.LayerNorm(d_model)

        self.head1 = nn.Sequential(nn.Linear(self.d_model, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 1)
                                  )
        
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)
        trunc_normal_(self.proj_patch, std=0.02)
        trunc_normal_(self.proj_classes, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        size_x = x.shape[1]
        masks_all=[]

        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        patches, cls_seg_feat = x[:, : -self.d_model], x[:, -self.d_model :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.head1(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        
        return masks