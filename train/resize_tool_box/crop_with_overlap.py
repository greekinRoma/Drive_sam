import math
import numpy as np
import numpy as np
import math
def reconstruct_from_grid_with_overlap(cropped_imgs, positions, original_shape, p):
    h, w, c = original_shape
    rows = math.ceil((h-p) / p)+1
    cols = math.ceil((w-p) / p)+1
    r_stride = (h-p)/(rows-1)
    h_stride = (w-p)/(cols-1)
    reconstructed = np.zeros((h, w, c), dtype=cropped_imgs[0].dtype)
    i = 0
    for row in range(rows):
        for col in range(cols):
            start_y = int(r_stride * row)
            start_x = int(h_stride * col)
            end_y = start_y + p
            end_x = start_x + p
            patch = cropped_imgs[i]
            i = i + 1
            tmp_mask = reconstructed[start_y:end_y, start_x:end_x]
            reconstructed[start_y:end_y, start_x:end_x] = np.bitwise_or(patch,tmp_mask)
            reconstructed[start_y:end_y, start_x:end_x] = np.bitwise_and(reconstructed[start_y:end_y, start_x:end_x],tmp_mask)
            reconstructed[start_y:end_y, start_x:end_x] = np.bitwise_and(reconstructed[start_y:end_y, start_x:end_x],patch)
                
    return reconstructed

def crop_image_grid_with_overlap(img,boxes,p):
    h, w, c = img.shape
    rows = math.ceil((h-p) / p)+1
    cols = math.ceil((w-p) / p)+1
    r_stride = (h-p)/(rows-1)
    h_stride = (w-p)/(cols-1)
    cropped_imgs = []
    new_boxes = []
    for row in range(rows):
        for col in range(cols):
            start_y = int(r_stride * row)
            start_x = int(h_stride * col)
            end_y = start_y + p
            end_x = start_x + p
            patch = img[start_y:end_y, start_x:end_x]
            block = np.zeros((p, p, c), dtype=img.dtype)
            block[0:end_y-start_y, 0:end_x-start_x] = patch
            
            cropped_imgs.append(block)
            new_boxes.append((0, 0, p, p))
    cropped_imgs = np.array(cropped_imgs)
    new_boxes = np.array(new_boxes)
    return cropped_imgs, new_boxes