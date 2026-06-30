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
    parser.add_argument("--onnx_output", type=str, default="segmentation_model.onnx")
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
    # 注意：训练时调用了 LORA(model=encoder.feature_encoder, rank=512)，这里也需要相同处理
    LORA(model=encoder.feature_encoder, rank=512)
    
    decoder = OurDecoder(
        args=args,
        in_channels=args.hidden_channels,
        out_channels=args.out_channels,
        fixed_size=args.fixed_size,
        scale_mode=args.scale_mode,
        decoder_mode=args.encoder_type,
        checkpoint=None
    )
    
    # 2. 加载训练好的权重
    # 由于训练时使用了 Accelerator 保存状态，需要用同样方式加载
    accelerator = Accelerator()
    # 先准备模型（但不需要优化器等）
    encoder, decoder = accelerator.prepare(encoder, decoder)
    # 加载状态（路径可以是 checkpoint 目录或包含 index 文件的目录）
    accelerator.load_state(args.checkpoint_path)
    # 获取原始未包装的模型
    encoder = accelerator.unwrap_model(encoder)
    decoder = accelerator.unwrap_model(decoder)
    
    model = SegmentationModel(encoder, decoder)
    model.to(args.device)
    model.eval()
    
    # 3. 创建示例输入
    batch_size = 1
    dummy_input = torch.randn(batch_size, args.in_channels, args.fixed_size, args.fixed_size, device=args.device)
    
    # 4. 导出 ONNX
    # 定义动态轴（可选）：批量大小和图像宽高可以动态变化
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_output,
        export_params=True,
        opset_version=19,          # 推荐使用 11 及以上
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"ONNX 模型已导出至: {args.onnx_output}")

if __name__ == "__main__":
    main()