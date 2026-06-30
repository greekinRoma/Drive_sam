import os
import cv2
import numpy as np

def replace_yellow_with_blue(input_dir, output_dir):
    """
    将 input_dir 下所有 mask 图像中的黄色 (255,255,0) 替换为蓝色 (0,0,255)
    保存为与原图相同格式的 RGB 图像
    """
    os.makedirs(output_dir, exist_ok=True)
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(extensions):
                continue
            src_path = os.path.join(root, file)
            img = cv2.imread(src_path)
            if img is None:
                print(f"无法读取: {src_path}")
                continue
            
            # BGR 格式下黄色为 (0, 255, 255)，蓝色为 (255, 0, 0)
            yellow_bgr = np.array([0, 255, 255], dtype=np.uint8)
            blue_bgr = np.array([255, 0, 0], dtype=np.uint8)
            
            # 找到黄色像素的位置
            mask = np.all(img == yellow_bgr, axis=-1)
            img[mask] = blue_bgr
            
            # 保持相对目录结构
            rel_path = os.path.relpath(src_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, img)
            print(f"已处理并保存: {out_path}")

if __name__ == "__main__":
    replace_yellow_with_blue("data/sundataset/test/masks", "./masks")