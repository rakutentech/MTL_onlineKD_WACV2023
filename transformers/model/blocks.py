
    
"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path

import torch.nn.functional as F

from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qk = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, y,mask=None):
        B, N, C = x.shape
        B, L, C = y.shape
        
        q = (
            self.qk(x)
            .reshape(B, N, self.heads, C // self.heads)
            .permute( 0, 2, 1, 3)
        )
        
        k = (
            self.qk(y)
            .reshape(B, L, self.heads, C // self.heads)
            .permute( 0, 2, 1, 3)
        )
        
        v = (
            self.qk(y)
            .reshape(B, L, self.heads, C // self.heads)
            .permute( 0, 2, 1, 3)
        )
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=0)
        attn = self.attn_drop(attn)
        
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn
    
class BlockTaskPatch(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, y, mask=None, return_attention=False):
        
        y, attn = self.attn(self.norm1(x), self.norm2(y), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttention1(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qk = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, y,mask=None):
        B, N, C = x.shape
        B, L, C = y.shape
        q = (
            self.qk(x)
            .reshape(B, N, self.heads, C // self.heads)
            .permute( 0, 2, 1, 3)
        )
        
        k = (
            self.qk(y)
            .reshape(B, L, self.heads, C // self.heads)
            .permute( 0, 2, 1, 3)
        )
        
        v = (
            self.qk(y)
            .reshape(B, L, self.heads, C // self.heads)
            .permute( 0, 2, 1, 3)
        )
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-2)
        attn = self.attn_drop(attn)
        
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn
 

 # class BlockTaskLayer(nn.Module):
#     def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#         self.norm4 = nn.LayerNorm(dim)
#         self.norm5 = nn.LayerNorm(dim)
#         self.attn = Attention(dim, heads, dropout)
#         self.crossattn = CrossAttention1(dim, heads, dropout)
#         self.mlp = FeedForward(dim, mlp_dim, dropout)
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
#     def attention(self, x, y, mask=None, return_attention=False):
#         y, attn = self.attn(self.norm1(x), self.norm2(y), mask) 
        
#         return attn

#     def forward(self, x, y, mask=None, return_attention=False):
#         x, attn = self.attn(self.norm1(x), mask)
#         y, attn = self.attn(self.norm2(y), mask)
#         x = x + self.drop_path(x)
#         x = self.norm1(x)
#         y = y + self.drop_path(y)
#         y = self.norm2(y)
#         x, attn = self.crossattn(self.norm3(x), self.norm4(y), mask)
#         if return_attention:
#             return attn
#         x = x + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm5(x)))
#         print(attn)
#         return x,y,attn

class BlockTaskLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.crossattn = CrossAttention1(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def attention(self, x, y, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), self.norm2(y), mask) 
        
        return attn

    def forward(self, x, y, mask=None, return_attention=False):

        x, attn = self.crossattn(self.norm3(x), self.norm4(y), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm5(x)))
        return x,y,attn