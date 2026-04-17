import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import *
from .positional_embedding import *


# Masked AutoEncoder 
class Modulator(nn.Module):
    
    def __init__(self, embed_dim=320, depth=4, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        
        super().__init__()
        
        # Class Token & Mask Token
        self.app_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mo_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Postional Embedding 
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, 320), requires_grad=False)
        
        # Block for Reconstruction
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        
        # Normalization
        self.norm = norm_layer(embed_dim)
        
        # Prediction Layer 
        self.pred = nn.Linear(embed_dim, embed_dim)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        
        # Intialization Positional Embedding 
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(1024**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Intialization Tokens
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.app_cls_token, std=.02)
        torch.nn.init.normal_(self.mo_cls_token, std=.02)
        
        # Initialization Layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # Random Masking Function
    def random_masking(self, x, mask_ratio):
       
        B, N, C = x.shape 
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device) 
        
        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
 
 
    def forward(self, app, mo, ratio=0.2):
        
        B, _, C = app.size()

        # Adjust masking ratio for Validation and Test
        ratio = 0.0 if B == 1 else ratio
        
        # Fused: BxHWxC
        x = torch.cat((app + self.pos_embed, mo + self.pos_embed), dim=1)
        
        # Random Masking for Fused Token 
        masked, mask, ids_restore = self.random_masking(x, ratio)
        
        # Append Mask Token & Restore Token Order
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - masked.shape[1], 1)
        x = torch.cat([masked, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))
        
        masked_fused = x.clone()
        
        # Append Class Token 
        app_cls_tokens = self.app_cls_token.expand(B, -1, -1)
        mo_cls_tokens = self.mo_cls_token.expand(B, -1, -1)                                   
        x = torch.cat((app_cls_tokens, mo_cls_tokens, x), dim=1)
        
        # Attention Operation
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # Reconstruction
        recon = self.pred(x)[:, 2:, :]

        return masked_fused, recon