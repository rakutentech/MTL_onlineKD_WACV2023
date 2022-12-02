"""
Adapted from 2020 Ross Wightmanload_weights
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn

from model.utils import init_weights, resize_pos_embed, load_weights
from model.blocks import Block, BlockTaskPatch, BlockTaskLayer, CrossAttention1

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

from timm.models.vision_transformer import _load_weights


        

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x
    
class Stage(nn.Module):
    def __init__(self, out_channels, layers):
        super(Stage, self).__init__()
        
        self.feature = nn.Sequential(layers)
        self.out_channels = out_channels
        
    def forward(self, x):
        return self.feature(x)

   
    
class VisionTransformer(nn.Module):
    def __init__(
        self,args,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3, 
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.task_n_layers = 2
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.distilled = distilled
        self.avgpool = nn.Sequential(
          nn.AdaptiveAvgPool1d((1))
         )
        embed_dim_attention =192; num_heads_attention = 3
        self.linear = nn.Linear(d_model, embed_dim_attention)
        self.linear_layer = nn.Linear((self.patch_embed.num_patches+
                                       len(args.tasks))*len(args.tasks)*embed_dim_attention,
                                      embed_dim_attention)
        self.linear_task = nn.Linear((self.patch_embed.num_patches+
                                      len(args.tasks))*n_layers*embed_dim_attention, 
                                     embed_dim_attention)
#         self.sig = nn.Sigmoid()
        
        # cls and pos tokens
        self.seg_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.depth_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sn_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + len(args.tasks), d_model))

        
        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.features = nn.Sequential(self.blocks)
        self.stages = [Stage(d_model,self.blocks[i]) for i in range(n_layers)]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.task_n_layers)]
      
        self.tasks = args.tasks
#         self.attn = Attention(dim, heads, dropout)
        self.weights = {}
        for i in args.tasks:           
            self.weights[i] = nn.Parameter(torch.rand(n_layers))
        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)
        self.norm1 = nn.LayerNorm(embed_dim_attention)
        self.norm2 = nn.LayerNorm(embed_dim_attention)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.seg_cls_token, std=0.02)
        trunc_normal_(self.depth_cls_token, std=0.02)
        trunc_normal_(self.sn_cls_token, std=0.02)
        self.pre_logits = nn.Identity()
#         self.uw = layer(device)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        load_weights(self, checkpoint_path, prefix)

#     def forward(self, im, return_features=False):
#         B, _, H, W = im.shape
#         PS = self.patch_size
        
#         x_seg=[];x_depth=[]
#         x = self.patch_embed(im)
        
#         if 'seg' in self.tasks: 
#             seg_cls_token = self.seg_cls_token.expand(B, -1, -1)
            
#             x = torch.cat((seg_cls_token, x), dim=1)
#         if 'depth' in self.tasks: 
#             depth_cls_token = self.depth_cls_token.expand(B, -1, -1)
#             x = torch.cat((depth_cls_token, x), dim=1)
#         if 'sn' in self.tasks: 
#             depth_cls_token = self.sn_cls_token.expand(B, -1, -1)
#             x = torch.cat((depth_cls_token, x), dim=1)
        
#         pos_embed = self.pos_embed


#         x = x + pos_embed
#         x = self.dropout(x)
#         features_seg=[]; features_depth=[];features_sn=[];total_features = []
        
#         x_seg = x;x_depth = x; x_sn = x
#         i=0
#         if 'seg' in self.tasks: 
#             for blk in self.blocks:
#                 x_seg = blk(x_seg) 
#                 features_seg.append(x_seg)
#                 i+=1
            
#             features = torch.stack(features_seg,dim=1) 
#             features = self.linear(features.squeeze(-1)).squeeze(-1)  
#             total_features.append(features)
#         i=0  
#         if 'depth' in self.tasks:
            
#             for blk in self.blocks:
#                 x_depth = blk(x_depth) 
#                 features_depth.append(x_depth)
#                 i+=1
            
#             features = torch.stack(features_depth,dim=1) 
#             features = self.linear(features.squeeze(-1)).squeeze(-1) 
#             total_features.append(features)
#         i=0  
#         if 'sn' in self.tasks: 
#             for blk in self.blocks:
#                 x_depth = blk(x_depth) 
#                 features_sn.append(x_sn)
#                 i+=1
            
#             features = torch.stack(features_sn,dim=1) 
#             features = self.linear(features.squeeze(-1)).squeeze(-1) 
#             total_features.append(features)
            


#         x_seg = self.norm(x_seg)
#         x_depth = self.norm(x_depth)
#         x_sn = self.norm(x_sn)
#         return x_seg, x_depth, x_sn
    
    def get_all_features(self, im): 
       
        B, _, H, W = im.shape
        PS = self.patch_size
        x_seg=[];x_depth=[]
        x = self.patch_embed(im)
        if 'segmentation' in self.tasks: 
            seg_cls_token = self.seg_cls_token.expand(B, -1, -1)          
            x = torch.cat((seg_cls_token, x), dim=1)
        if 'depth' in self.tasks: 
            depth_cls_token = self.depth_cls_token.expand(B, -1, -1)
            x = torch.cat((depth_cls_token, x), dim=1)
        if 'normal' in self.tasks: 
            depth_cls_token = self.depth_cls_token.expand(B, -1, -1)
            x = torch.cat((depth_cls_token, x), dim=1)
            
        pos_embed = self.pos_embed
        
        x = x + pos_embed
        x = self.dropout(x)
        features = []
        for blk in self.blocks:
            features.append(blk(x))
        x = torch.stack(features)
        
        return x
        
    def get_final_features(self, im): 
        B, _, H, W = im.shape
        PS = self.patch_size
        x_seg=[];x_depth=[]
        x = self.patch_embed(im)
        if 'segmentation' in self.tasks: 
            seg_cls_token = self.seg_cls_token.expand(B, -1, -1)          
            x = torch.cat((seg_cls_token, x), dim=1)
        if 'depth' in self.tasks: 
            depth_cls_token = self.depth_cls_token.expand(B, -1, -1)
            x = torch.cat((depth_cls_token, x), dim=1)
        if 'normal' in self.tasks: 
            depth_cls_token = self.depth_cls_token.expand(B, -1, -1)
            x = torch.cat((depth_cls_token, x), dim=1)
            
        pos_embed = self.pos_embed
        
        x = x + pos_embed
        x = self.dropout(x)
        for blk in self.blocks:
            x= blk(x)
        
        
        return x
    
    
    
    def return_layer_features(self, im):       
        B, _, H, W = im.shape
        PS = self.patch_size
        x_seg=[];x_depth=[]
        x = self.patch_embed(im)
        if 'seg' in self.tasks: 
            seg_cls_token = self.seg_cls_token.expand(B, -1, -1)          
            x = torch.cat((seg_cls_token, x), dim=1)
        if 'depth' in self.tasks: 
            depth_cls_token = self.depth_cls_token.expand(B, -1, -1)
            x = torch.cat((depth_cls_token, x), dim=1)
        if 'sn' in self.tasks: 
            depth_cls_token = self.depth_cls_token.expand(B, -1, -1)
            x = torch.cat((depth_cls_token, x), dim=1)
            
        pos_embed = self.pos_embed
        
        x = x + pos_embed
        x = self.dropout(x)
        features_seg=[]; features_depth=[];features_sn=[];total_features = []
        
        x_seg = x;x_depth = x; x_sn = x
        i=0
        
        if 'seg' in self.tasks: 
            for blk in self.blocks:
                x_seg = blk(x_seg)  
                features_seg.append(x_seg)
                i+=1
        
        if 'depth' in self.tasks: 
           
            for blk in self.blocks:
                x_depth = blk(x_depth) 
                features_depth.append(x_depth)
               
                i+=1 
        if 'sn' in self.tasks: 
            for blk in self.blocks:
                x_sn = blk(x_sn) 
                features_sn.append(x_sn)
                i+=1
#         print("segment feature shape inside vit.py",self.tasks,len(features_seg))
        return features_seg, features_depth, features_sn
    
    def get_next_features(self, x, index):     
        i=0
        for blk in self.blocks:
            if i==index:
                x = blk(x) 
            i+=1
        return x
    
    def decoder_attention(self, x,y):
        weights={}
        x, attention = self.crossattn(self.norm1(x), self.norm2(y), None) 
        attention = nn.functional.softmax(attention.mean(dim=1)[0,:],dim=0)
        if 'seg' in self.tasks:  
            ind = self.tasks.index('seg')   
            weights['seg'] = attention[ind,:]#[:,:,ind,:].mean(dim=1)[0,:]
        if 'depth' in self.tasks:
            ind = self.tasks.index('depth')     
            weights['depth'] = attention[ind,:]#attention[:,:,ind,:].mean(dim=1)[0,:]
        if 'sn' in self.tasks:
            ind = self.tasks.index('sn')     
            weights['sn'] = attention[ind,:]#attention[:,:,ind,:].mean(dim=1)[0,:]
        return weights


    
    def get_attention_map(self, im, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
