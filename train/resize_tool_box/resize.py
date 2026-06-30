import math
import numpy as np
import cv2
def reconstruct_from_resize(resized_imgs, positions, original_shape, p):
    h, w, c = original_shape
    reconstructed=cv2.resize(resized_imgs[0],(w,h))
    reconstructed = np.expand_dims(reconstructed,axis=-1)
    reconstructed = np.repeat(reconstructed,axis=-1,repeats=3)
    return reconstructed

def resize_img(img,boxes,p):
    
    cropped_imgs = [cv2.resize(img,(p,p))]
    new_boxes = [(0, 0, p, p)]
    
    cropped_imgs = np.array(cropped_imgs)
    new_boxes = np.array(new_boxes)
    return cropped_imgs, new_boxes