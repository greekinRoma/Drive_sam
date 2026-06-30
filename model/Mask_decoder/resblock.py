import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels=192, out_channels=256):
        super().__init__()
        mid_channels = out_channels
        if mid_channels < 1:
            mid_channels = 1

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,bias=False)
        self.norm1 = LayerNorm2d(mid_channels)   # 替换原 nn.LayerNorm()
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.norm2 = LayerNorm2d(out_channels)   # 替换原 nn.LayerNorm() 或 nn.BatchNorm2d

        # 可选：添加残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out