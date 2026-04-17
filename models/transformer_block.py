import torch
import torch.nn as nn

class Mlp(nn.Module):

    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 bias=True
                 ):
        
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        
        return x


class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 16,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            norm_layer: nn.Module = nn.LayerNorm
            ) -> None:
        
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
                   
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        
        B, N, C = x.shape                                                                                      
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)                                                                          

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp
            ) -> None:
        
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, norm_layer=norm_layer)
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Create Attention Mask (default: 0 & Calculatable: 1)
        B, total_tokens, _ = x.size()
        img_cls_idx, motion_cls_idx = 0, 1
        
        mask = torch.zeros(B, total_tokens, total_tokens, device=x.device)  
        
        mask[:, img_cls_idx, 2:1026] = 1
        mask[:, motion_cls_idx, 1026:] = 1
        mask[:, 2:, :] = 1
        
        for i in range(2):
            mask[:,i,i] = 1
        
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    
