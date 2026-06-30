import torch
from torch import nn
import torch.nn.functional as F
from .BDMv1.Decomposition_and_Reconstruction import ReconstructionModel as ReRemodelv1
from .BDMv2.Decomposition_and_Reconstruction import ReconstructionModel as ReRemodelv2
from .upsample import Upsample4xSequential
class Scalar_Reconstruction(nn.Module):
    """
    支持单个 3D (C, H, W) 或 4D (N, C, H, W) 张量的两步骤缩放模块：
        - adjust(tensor): 将输入张量缩放到固定尺寸，并记录原始空间尺寸
        - restore(tensor): 将缩放后的张量还原到记录的原始尺寸
    支持插值模式: 'bilinear' 或 'nearest'
    """
    def __init__(self, in_channels, out_channels, fixed_size, mode='bilinear',use_outconv=False):
        super().__init__()
        self.fixed_size = fixed_size
        self.use_outconv = use_outconv
        mode = mode.strip("_hq")
        self.mode = mode
        self.align_corners = (mode == 'bilinear')
        self.original_size = None  # 存储原始空间尺寸 (H, W)
        if mode == 'bilinearconv' or mode == "nearestconv":
           
            out_dim = in_channels // 16
        elif mode == 'bilinear' or mode == "nearest":
            out_dim = in_channels
        elif mode == "BDMv1":
            self.BDM_scalar = ReRemodelv1(m=256,n=256,num_head=1,origin_patch_size=16,resize_patch_size=16,resize_embed_dim=in_channels,origin_embed_dim=in_channels)
            out_dim = in_channels // 64
        elif mode == "BDMv2":
            self.BDM_scalar = ReRemodelv2(resize_embed_dim=256,origin_embed_dim=256)
            out_dim = in_channels // 64
        self.out_conv = nn.Conv2d(in_channels=out_dim,out_channels=out_channels,kernel_size=1,stride=1) 

    def forward(self, x, original_size):
        if original_size is None:
            raise RuntimeError("Must call adjust() first to record original size.")
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D tensor, but got {x.dim()}D tensor")
        
        if self.mode == "bilinear" or self.mode == "nearest":
            if self.use_outconv:
                restored = self.out_conv(x.unsqueeze(0) if x.dim() == 3 else x)
            else:
                restored = x.unsqueeze(0) if x.dim() == 3 else x
            restored = F.interpolate(
                        restored,
                        size=original_size,
                        mode=self.mode,
                        align_corners=self.align_corners
                    )
            restored = self.out_conv(restored)
        elif self.mode == 'bilinearconv' or self.mode == "nearestconv":
            restored = x.unsqueeze(0) if x.dim() == 3 else x
            if self.use_outconv:
                restored = self.out_conv(restored)
            restored = F.interpolate(
                        restored,
                        size=original_size,
                        mode=self.mode.replace("conv", ""),
                        align_corners=self.align_corners
                    )
        elif self.mode == "BDMv1":
            restored = self.BDM_scalar.decode(patch=x.unsqueeze(0) if x.dim() == 3 else x, attn=self.attn, pad_size=self.pad_size, orgin_size=self.orgin_size)
            if self.use_outconv:
                restored = self.out_conv(restored)
        elif self.mode == "BDMv2":
            restored = self.BDM_scalar.decode(patch=x.unsqueeze(0) if x.dim() == 3 else x, attn=self.attn, pad_size=self.pad_size, orgin_size=self.orgin_size)
            if self.use_outconv:
                restored = self.out_conv(restored)
        return restored