import torch
import torch.nn as nn
from einops import rearrange
class Tiny_attention(nn.Module):
    def __init__(self, dim=768, num_head=8):
        super().__init__()
        self.num_head = num_head
        self.dim = dim
        self.qvk = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        qvk = self.qvk(x)
        q, v, k = rearrange(
            qvk, "b l (k h d) -> k b h l d", k=3, h=self.num_head
        ).unbind(0)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.proj(x)
        return x