import torch
import torch.nn as nn

class Upsample4xSequential(nn.Module):
    """
    使用 4 次双线性上采样（每次 scale_factor=2）将特征图放大 16 倍，
    每次上采样后都通过深度可分离卷积块将通道数减半。

    通道变化规律：
        - 输入: in_channels
        - 第1次上采样后: in_channels // 2
        - 第2次上采样后: in_channels // 4
        - 第3次上采样后: in_channels // 8
        - 第4次上采样后: in_channels // 16

    若 in_channels 不是 16 的倍数，会向下取整（但至少为 1）。

    Args:
        in_channels (int): 输入通道数（必须 >= 16，否则最终通道会变成0或1，失去意义）。
        min_channels (int, optional): 允许的最小通道数，默认为 1。
    """
    def __init__(self, in_channels, min_channels=1, scale_factor=2, num_layers=4):
        super().__init__()
        layers = []
        curr_ch = in_channels

        for i in range(num_layers):  # 4次上采样
            next_ch = max(curr_ch // 2, min_channels)
            layers.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True))
            layers.append(self._make_dconv_block(curr_ch, next_ch))
            curr_ch = next_ch

        self.seq = nn.Sequential(*layers)

    def _make_dconv_block(self, in_ch, out_ch):
        """深度可分离卷积块：Depthwise Conv + Pointwise Conv + BN + ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),  # depthwise
            nn.Conv2d(in_ch, out_ch, kernel_size=1),                           # pointwise
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


# ========== 测试代码 ==========
if __name__ == "__main__":
    x = torch.randn(1, 64, 16, 16)   # 输入 64 通道，16x16
    model = Upsample4xHalvingChannels(in_channels=64, min_channels=1)
    y = model(x)

    print(f"输入形状: {x.shape}")      # torch.Size([1, 64, 16, 16])
    print(f"输出形状: {y.shape}")      # torch.Size([1, 4, 256, 256]) 因为 64 → 32 → 16 → 8 → 4
    print("每次上采样后通道数: 64 -> 32 -> 16 -> 8 -> 4，严格减半。")