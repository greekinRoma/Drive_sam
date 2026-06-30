import math
import numpy as np
def crop_images(img, boxes, p):
    """
    Crop the image based on the given bounding box and ensure that each target is centered.
    
    args:
    - img: Original image size, shape is [h, w, 3]。
    - boxes: Bounding box coordinates, shape is  [N, 4], format is (x1, y1, x2, y2)。
    - p: crop size 。
    
    return:
    - cropped_imgs: The cropped array of images, shape is [N, p, p, 3]。
    - updated_boxes: Updated bounding box coordinates, shape is [N, 4]。
    """
    img = img
    boxes = boxes
    h, w, _ = img.shape
    cropped_imgs = []
    new_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # Calculate the start and end points for cropping
        start_x = max(int(center_x - p // 2), 0)
        start_y = max(int(center_y - p // 2), 0)
        end_x = min(int(center_x + p // 2), w)
        end_y = min(int(center_y + p // 2), h)

        # Crop the image
        cropped_img = img[start_y:end_y, start_x:end_x, :]

        # Pad the cropped image to the desired size if necessary
        pad_height = max(0, p - cropped_img.shape[0])
        pad_width = max(0, p - cropped_img.shape[1])
        if pad_height > 0 or pad_width > 0:
            cropped_img = np.pad(cropped_img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

        cropped_imgs.append(cropped_img)

        # Update the box coordinates after padding
        pad_left = (p - cropped_img.shape[1]) // 2
        pad_top = (p - cropped_img.shape[0]) // 2
        new_x1 = max(0, (x1 - start_x) + pad_left)
        new_y1 = max(0, (y1 - start_y) + pad_top)
        new_x2 = min(p, (x2 - start_x) + pad_left)
        new_y2 = min(p, (y2 - start_y) + pad_top)
        new_boxes.append([new_x1, new_y1, new_x2, new_y2])
    cropped_imgs = np.array(cropped_imgs)
    new_boxes = np.array(new_boxes)
    return cropped_imgs, new_boxes


def reconstruct_image(cropped_imgs,original_boxes, original_shape, p):


    h, w, _ = original_shape
    reconstructed_img = np.zeros((h, w, 1), dtype=np.uint8) 
    boxes = original_boxes
    for i, cropped_img in enumerate(cropped_imgs):
        box = boxes[i]
        x1, y1, x2, y2 = box
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        start_x = int(center_x - p // 2)
        start_y = int(center_y - p // 2)
        if start_x > w or start_y > h:
            continue
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(start_x + p, w)
        end_y = min(start_y + p, h)

        crop_width = end_x - start_x
        crop_height = end_y - start_y
        
        mask = (cropped_img > 0).any(axis=-1)  
        reconstructed_img[start_y:end_y, start_x:end_x, :][mask[:crop_height, :crop_width]] = cropped_img[:crop_height, :crop_width][mask[:crop_height, :crop_width]]
  
    return reconstructed_img