import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from efficient_sam.build_efficient_sam import build_efficient_sam_vits
import os

def segment_everything(image_path, output_dir="results", grid_size=32):
    """
    使用 EfficientSAM 分割图片中的所有物体
    :param image_path: 输入图片路径
    :param output_dir: 结果保存目录
    :param grid_size: 网格采样密度（越大分割越细，但速度越慢）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = build_efficient_sam_vits().to(device).eval()
    
    # 2. 预处理图片
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return
        
    image_pil = Image.open(image_path).convert("RGB")
    w, h = image_pil.size
    image_tensor = transforms.ToTensor()(image_pil).to(device).unsqueeze(0)

    # 3. 生成网格点 (Segment Everything 的关键)
    # 在图像上均匀分布 grid_size x grid_size 个点
    x = np.linspace(0, w, grid_size)
    y = np.linspace(0, h, grid_size)
    xv, yv = np.meshgrid(x, y)
    points = np.stack([xv.flatten(), yv.flatten()], axis=-1)
    
    # 将点转换为模型需要的格式 [Batch, num_queries, num_points_per_query, 2]
    # 这里我们将每个点作为一个独立的 query 处理
    input_points = torch.tensor(points).reshape(1, points.shape[0], 1, 2).to(device).float()
    input_labels = torch.ones((1, points.shape[0], 1)).to(device).int()

    # 4. 推理
    print(f"正在处理: {image_path}，采样点数: {points.shape[0]}...")
    with torch.inference_mode():
        # EfficientSAM 能够并行处理大量 queries
        predicted_logits, predicted_iou = model(
            image_tensor,
            input_points,
            input_labels,
        )

    # 5. 后处理与合并
    # 提取每个点预测出的最高质量 mask (index 0)
    # 形状: [1, num_queries, 3, H, W] -> 取每组 query 的第一个候选
    masks = (predicted_logits[0, :, 0, :, :] > 0).cpu().numpy()
    
    # 创建一个彩色的分割图
    result_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.any():
            # 给每个 mask 分配一个随机颜色
            color = np.random.randint(0, 255, (3,)).tolist()
            result_mask[m] = color

    # 6. 保存结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(output_dir, f"{base_name}_segmented.png")
    
    # 将原图与遮罩混合显示 (50% 透明度)
    combined = Image.blend(image_pil, Image.fromarray(result_mask), alpha=0.5)
    combined.save(save_path)
    print(f"结果已保存至: {save_path}")

# 使用示例
if __name__ == "__main__":
    # 你可以在这里指定任何图片路径
    my_image = "figs/examples/dogs.jpg" 
    segment_everything(my_image, grid_size=30)