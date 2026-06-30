import torch
from torch import nn
import torch.nn.functional as F
from .BDMv1.Decomposition_and_Reconstruction import DecompositionModel as DeRemodelv1
from .BDMv2.Decomposition_and_Reconstruction import DecompositionModel as DeRemodelv2
from .upsample import Upsample4xSequential
class Scalar_Decomposition(nn.Module):
    """
    支持单个 3D (C, H, W) 或 4D (N, C, H, W) 张量的两步骤缩放模块：
        - adjust(tensor): 将输入张量缩放到固定尺寸，并记录原始空间尺寸
        - restore(tensor): 将缩放后的张量还原到记录的原始尺寸
    支持插值模式: 'bilinear' 或 'nearest'
    """
    def __init__(self, in_channels, out_channels, fixed_size, mode='bilinear'):
        super().__init__()
        self.fixed_size = fixed_size
        self.mode = mode
        self.align_corners = (mode == 'bilinear')
        self.original_size = None  # 存储原始空间尺寸 (H, W)
        mode = mode.strip("_hq")
        self.mode = mode
        if mode == 'bilinearconv' or mode == "nearestconv":
            self.upsample = Upsample4xSequential(in_channels=in_channels)
            out_dim = in_channels // 16
        elif mode == 'bilinear' or mode == "nearest":
            out_dim = in_channels
        elif mode == "BDMv1":
            self.BDM_scalar = DeRemodelv1(m=256,n=256,num_head=1,origin_patch_size=16,resize_patch_size=16,resize_embed_dim=in_channels,origin_embed_dim=in_channels)
            out_dim = in_channels // 64
        elif mode == "BDMv2":
            self.BDM_scalar = DeRemodelv2(resize_embed_dim=256,origin_embed_dim=256)
            out_dim = in_channels // 64
        self.scale_conv = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels=out_dim,out_channels=out_channels,kernel_size=1,stride=1) 

    def forward(self, x):
        # print(x.shape)
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D tensor, but got {x.dim()}D tensor")
        
        # 获取空间尺寸
        if x.dim() == 3:
            h, w = x.shape[1], x.shape[2] 
        else:  # 4D
            h, w = x.shape[2], x.shape[3]
        x = x.unsqueeze(0) if x.dim() == 3 else x
        if self.mode == "bilinear" or self.mode == "nearest":
            scaled = F.interpolate(
                x,  # 3D 需要加 batch 维
                size=self.fixed_size,
                mode=self.mode,
                align_corners=self.align_corners
            )
        elif self.mode == 'bilinearconv' or self.mode == "nearestconv":
            scaled = F.interpolate(
                x,  # 3D 需要加 batch 维
                size=self.fixed_size,
                mode=self.mode.replace("conv", ""),
                align_corners=self.align_corners
            )
        elif self.mode == "BDMv1":
            scaled, self.attn, self.pad_size, self.orgin_size = self.BDM_scalar.encode(x)
        elif self.mode == "BDMv2":
            scaled, self.attn, self.pad_size, self.orgin_size = self.BDM_scalar.encode(x)
        return scaled