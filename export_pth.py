import torch
import argparse
from model.LoRA_encoder.lora_encoder import OurEncoder, LORA
from model.Mask_decoder.mask_decoder import OurDecoder
from accelerate import Accelerator
from torch import nn
class SegmentationModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.act = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type", type=str, default="efficient_sam_vitt")
    parser.add_argument("--decoder_type", type=str, default="Simple")
    parser.add_argument("--scale_mode", type=str, default="bilinearconv")
    parser.add_argument("--fixed_size", type=int, default=1024)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--decoder_use_outconv", type=bool, default=True)
    parser.add_argument("--checkpoint_path", type=str,
                        default="result/drive_sam/efficient_sam_vitt/Simple/bilinearconv/checkpoint/checkpoint_99")
    parser.add_argument("--pth_output", type=str, default="segmentation_model.pth")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. 构建模型结构（与训练时完全一致）
    encoder = OurEncoder(
        args=args,
        in_channels=args.in_channels,
        out_channels=args.hidden_channels,
        fixed_size=args.fixed_size,
        scale_mode=args.scale_mode,
        encoder_type=args.encoder_type
    )
    LORA(model=encoder.feature_encoder, rank=512)   # 必须与训练时的 LoRA 配置一致
    
    decoder = OurDecoder(
        args=args,
        in_channels=args.hidden_channels,
        out_channels=args.out_channels,
        fixed_size=args.fixed_size,
        scale_mode=args.scale_mode,
        decoder_mode=args.encoder_type,
        checkpoint=None
    )
    
    # 2. 加载 Accelerator 保存的完整状态
    accelerator = Accelerator()
    encoder, decoder = accelerator.prepare(encoder, decoder)
    accelerator.load_state(args.checkpoint_path)      # 加载权重
    encoder = accelerator.unwrap_model(encoder)
    decoder = accelerator.unwrap_model(decoder)
    
    model = SegmentationModel(encoder, decoder)
    model.to(args.device)
    model.eval()
    
    # 3. 保存为 .pth 文件（仅模型权重）
    torch.save(model.state_dict(), args.pth_output)
    print(f"模型权重已保存至: {args.pth_output}")

if __name__ == "__main__":
    main()