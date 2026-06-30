import math
import numpy as np
def reconstruct_from_grid(cropped_imgs, positions, original_shape, p):
    """
    从网格裁剪的图块重建原始图像。

    args:
    - cropped_imgs: 列表，每个元素为 p×p×3 的裁剪图块
    - positions: 列表，每个元素为 (start_x, start_y) 图块左上角在原图中的坐标
    - original_shape: 原始图像形状 (h, w, 3)
    - p: 裁剪尺寸

    returns:
    - reconstructed: 重建图像，shape (h, w, 3)
    """
    h, w, c = original_shape
    rows = math.ceil(h / p)
    cols = math.ceil(w / p)
    reconstructed = np.zeros((h, w, c), dtype=cropped_imgs[0].dtype)
    i = 0
    for row in range(rows):
        for col in range(cols):
            start_y = row * p
            start_x = col * p
            orig_w = min(p, w - start_x)
            orig_h = min(p, h - start_y)
            # 从图块中提取有效部分
            block = cropped_imgs[i][:orig_h, :orig_w]
            i = i + 1
            # 放回原图对应位置
            reconstructed[start_y:start_y+orig_h, start_x:start_x+orig_w] = block
                
    return reconstructed

def crop_image_grid(img,boxes,p):
    """
    将图像划分为 p×p 的网格，从左到右、从上到下依次裁剪，
    边缘不足时自动补零。

    args:
    - img: 原始图像，shape [h, w, 3]
    - p: 裁剪尺寸

    returns:
    - cropped_imgs: 列表，每个元素为 p×p×3 的裁剪图块
    - positions: 列表，每个元素为 (start_x, start_y) 图块左上角在原图中的坐标
    """
    h, w, c = img.shape
    rows = math.ceil(h / p)
    cols = math.ceil(w / p)
    
    cropped_imgs = []
    new_boxes = []
    
    for row in range(rows):
        for col in range(cols):
            start_y = row * p
            start_x = col * p
            end_y = min(start_y + p, h)
            end_x = min(start_x + p, w)
            
            # 裁剪实际区域
            patch = img[start_y:end_y, start_x:end_x]
            
            # 创建 p×p 的零矩阵，将实际区域放入左上角
            block = np.zeros((p, p, c), dtype=img.dtype)
            block[0:end_y-start_y, 0:end_x-start_x] = patch
            
            cropped_imgs.append(block)
            new_boxes.append((0, 0, p, p))
    cropped_imgs = np.array(cropped_imgs)
    new_boxes = np.array(new_boxes)
    return cropped_imgs, new_boxes