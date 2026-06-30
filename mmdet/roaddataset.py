import torch.utils.data.dataset as Dataset
import numpy as np
import torch
import os 
import skimage.io as skio
from mmdet.datasets.transforms.transforms import *
from mmcv.transforms import RandomResize,CenterCrop
import random
class Road_Dataset(Dataset.Dataset):
    def __init__(self, datasetroot,split,use_argumentation):
        self.imgpath = os.path.join(datasetroot,'images',split)
        self.onehotpath = os.path.join(datasetroot,'masks',split)
        self.img = os.listdir(self.imgpath)
        self.img = random.sample(self.img,len(self.img))
        self.use_argumentation = use_argumentation
        self.transRandomResize = RandomResize(scale=(1024,1024),ratio_range=(0.5, 2.0),resize_type='Resize',keep_ratio=True)
        self.transRandomCrop =RandomCrop(crop_size=(1024,1024),crop_type='absolute',allow_negative_crop =True,recompute_bbox=True)
        self.transFixShapeResize = FixShapeResize(width=1024,height=1024,pad_val=dict(img=144, seg=0),keep_ratio=True)
        self.transRandomFlip = RandomFlip(0.1,'diagonal')
        self.transPhotoMetricDistortion= PhotoMetricDistortion()
        self.transPad = Pad(size=(1024,1024),pad_val=dict(img=144, seg=0))
        self.train_mode = split
    def __len__(self):
        return len(self.img)
    def __getitem__(self, index):
        sample ={}
        imagename = self.img[index]
        img  = skio.imread(os.path.join(self.imgpath,self.img[index]))
        onehotmask = np.load(os.path.join(self.onehotpath,self.img[index].replace('.jpg','.npy')))
        sample = {'img':img, 'gt_seg_map':onehotmask}    
        if self.train_mode == 'train': 
            if self.use_argumentation == True:   
                sample = self.transRandomResize(sample)
                sample = self.transRandomCrop(sample)
                sample = self.transRandomFlip(sample)           
                sample = self.transPhotoMetricDistortion(sample)
        img = sample['img']
        onehotmask = sample['gt_seg_map']
        img = torch.tensor(img.copy(), dtype=torch.float32).permute(2,0,1)
        onehotmask = torch.tensor(onehotmask.copy(), dtype=torch.float32).permute(2,0,1)
        sample ={'img':img,'onehotmask':torch.clip(onehotmask,max=1.,min=0.), 'filename':imagename}
        return  sample