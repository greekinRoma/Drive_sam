from mmdet.sundronedataset import SUNDRONE_Dataset
import cv2
import numpy as np
import torch

def visualize_dataset():
    # 1. 初始化数据集
    # 建议将数据路径设为变量
    dataset_root = "data/sundataset"
    sundrone_dataset = SUNDRONE_Dataset(
        datasetroot=dataset_root,
        split='train',
        use_argumentation=True
    )
    
    print(f"Dataset loaded. Size: {len(sundrone_dataset)}")
    print("Press 'any key' for next sample, 'q' to quit.")

    # 2. 开启窗口优化
    cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL) 

    # 3. 不断循环
    try:
        while True:
            # 随机获取一个索引
            idx = np.random.randint(0, len(sundrone_dataset))
            sample = sundrone_dataset[idx]
            
            # --- 数据转换 ---
            # 从 Tensor [C, H, W] 转回 Numpy [H, W, C]
            # 使用 .detach().cpu() 确保在各种环境下都能运行
            img = sample['img'].permute(1, 2, 0).detach().cpu().numpy()
            mask = sample['onehotmask'].permute(1, 2, 0).detach().cpu().numpy()
            
            # --- 归一化与类型修正 ---
            # 如果 img 是 float32 且范围在 0-255，imshow 需要除以 255
            # 如果 img 已经是 0-1，直接显示
            show_img = img / 255.0 if img.max() > 1.0 else img
            # 确保是 BGR 顺序（如果原图是 RGB，cv2 需要转换）
            show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)

            # Mask 处理：如果是 One-hot 或多通道，取前三通道或转为灰度
            # 如果 mask 只有 0 和 1，乘以 255 方便观察
            show_mask = mask.astype(np.float32)
            if show_mask.max() <= 1.0:
                show_mask *= 255.0
            show_mask = show_mask.astype(np.uint8)

            # --- 拼图显示 (横向拼接原图和掩码) ---
            # 确保高度一致
            combined = np.hstack((show_img * 255.0, show_mask.astype(np.float32)))
            combined = combined.astype(np.uint8)

            # 绘制信息描述
            cv2.putText(combined, f"Name: {sample['filename']} | Idx: {idx}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 4. 显示与交互
            cv2.imshow("Visualization", combined)
            
            # 等待按键，10ms 刷新一次。如果按下 'q' 键则退出
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27: # 27 是 Esc
                break
                
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Test finished.")

if __name__ == "__main__":
    visualize_dataset()