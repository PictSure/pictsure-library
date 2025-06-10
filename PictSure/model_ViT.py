import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # e.g. 14 if 224 // 16
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, H, W]
        # project to embeddings with shape [B, D, #patches_row, #patches_col]
        x = self.proj(x)  # -> [B, embed_dim, grid_size, grid_size]
        # flatten the spatial dims
        x = x.flatten(2)  # -> [B, embed_dim, grid_size*grid_size]
        x = x.transpose(1, 2)  # -> [B, #patches, embed_dim]
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # -> [B, N, 3*C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # -> [3, B, heads, N, C//heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # scaled dot product
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim, hidden_features=int(dim*mlp_ratio),
            out_features=dim, drop=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 1D positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        # simple initialization
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.normal_(self.head.bias, std=1e-6)

    def forward(self, x):
        # x shape: [B, 3, H, W]
        B = x.shape[0]
        x = self.patch_embed(x)  # -> [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # -> [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # -> [B, N+1, D]

        x = x + self.pos_embed[:, :(x.size(1)), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # extract CLS token
        cls_token_final = x[:, 0]
        # classification
        logits = self.head(cls_token_final)

        return logits, cls_token_final
