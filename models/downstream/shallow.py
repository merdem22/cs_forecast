# models/downstream/shallow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ShallowStage1(nn.Module):
    """
    Approximation of the Shallow ConvNet (temporal+spatial conv, square, mean pool, log).
    Designed for cybersickness detection (Stage 1) with binary sigmoid output.
    """

    def __init__(
        self,
        n_channels: int = 14,
        n_outputs: int = 1,
        kernel_len: int = 25,
        F1: int = 40,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.temporal = nn.Conv2d(1, F1, (1, kernel_len), bias=False)
        # Depthwise spatial conv across channels
        self.spatial = nn.Conv2d(F1, F1, (n_channels, 1), groups=F1, bias=False)
        self.bn = nn.BatchNorm2d(F1)
        self.pool_size = int(pool_size)
        self.pool_stride = int(pool_stride)
        self.drop = nn.Dropout(dropout)
        # 1x1 conv head; we average over time dimension before sigmoid
        self.classifier = nn.Conv2d(F1, n_outputs, kernel_size=(1, 1))
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C, T)
        B, W, C, T = x.shape
        h = rearrange(x, "B W C T -> (B W) 1 C T")
        h = self.temporal(h)
        h = self.spatial(h)
        h = self.bn(h)
        h = torch.square(h)
        # Clamp pool kernel/stride to current temporal length to avoid edge cases
        t_len = h.shape[-1]
        pool_k = min(self.pool_size, t_len)
        pool_s = min(self.pool_stride, pool_k)
        h = F.avg_pool2d(h, kernel_size=(1, pool_k), stride=(1, pool_s))
        h = torch.log(torch.clamp(h, min=1e-6))
        h = self.drop(h)
        h = self.classifier(h)  # (B*W, n_outputs, 1, T')
        h = h.mean(dim=-1).squeeze(-1)  # average over time -> (B*W, n_outputs)
        h = rearrange(h, "(B W) O -> B W O", B=B, W=W)
        h = h.mean(dim=1)  # aggregate across windows
        return self.out_act(h)


__all__ = ["ShallowStage1"]
