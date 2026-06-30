import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from typing import  List, Tuple
from train.segment_anything_training import sam_model_registry
from train.segment_anything_training.modeling import TwoWayTransformer, MaskDecoder
from torch.utils.data import DataLoader
from train.utils.loss_mask import loss_masks, loss_masks_no_sample
from train.utils.loss_bce import bce_dice
import train.utils.misc as misc
import skimage.io as skio
import numpy as np
from train.mmdet.rossamdataset import ROSSAM_Dataset
from safetensors import safe_open
import math
from tqdm import tqdm
from safetensors.torch import save_file
import warnings
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs,InitProcessGroupKwargs
from accelerate.utils import set_seed
from mmdet.sundronedataset import SUNDRONE_Dataset
from mmdet.roaddataset import Road_Dataset
from model.LoRA_encoder.lora_encoder import OurEncoder, LORA
from model.Mask_decoder.mask_decoder import OurDecoder
from train.utils.loss_mask import compute_metrics
def custom_collate(batch):
    # batch 是一个 list，每个元素是 __getitem__ 返回的 dict
    imgs = [item['img'] for item in batch]
    masks = [item['onehotmask'] for item in batch]
    filenames = [item['filename'] for item in batch]
    # 不 stack，直接返回 list of tensors
    return {'img': imgs, 'onehotmask': masks, 'filename': filenames}
def train(args, encoder, optimizer_encoder, decoder, optimizer_decoder, train_dataloaders, valid_dataloaders, lr_scheduler, accelerator):
    best_iou_epoch = 0
    best_iou = 0.
    best_dice_iou = 0.
    output_dir = os.path.join(args.output,args.encoder_type, args.decoder_type, args.scale_mode)
    out_channels = args.out_channels
    loss_bce_dice = bce_dice(args=args)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    for epoch in range(epoch_start,epoch_num): 
        allloss = 0
        epoch_dir = os.path.join(output_dir,"val_images",f"epoch_{epoch}")
        os.makedirs(epoch_dir,exist_ok=True)
        for iteridx,data in enumerate(tqdm(train_dataloaders,disable=not accelerator.is_local_main_process,miniters=20 )):
            inputs  = data['img']
            onehotmasks = data['onehotmask']
            
            image_encoding_results = encoder(inputs)
            results = decoder(image_encoding_results)
            losses = []
            for result, onehotmask in zip(results, onehotmasks):
                loss = loss_bce_dice(y_pred=result, y_true=onehotmask)
                losses.append(loss)
            loss = sum(losses)/len(losses)
            optimizer_decoder.zero_grad()
            optimizer_encoder.zero_grad()
            accelerator.backward(loss)
            optimizer_encoder.step()
            optimizer_decoder.step()
            allloss += float(loss.detach().cpu().numpy())

        avg_loss = allloss/(iteridx+1)
        lr_scheduler.step()        
        with open(os.path.join(output_dir,'train_stats.txt'), 'a') as file:
            file.write(f'Epoch {epoch} stats:\n')  
            file.write(f'training_loss: {avg_loss:.4f} ]\n') 
        accelerator.save_state(os.path.join(output_dir,"checkpoint",f'checkpoint_{epoch}')) 


        total_iou = 0.0
        total_dice = 0.0
        total_pa = 0.0
        num_samples =0
        
        for iteridx,data in enumerate(tqdm(valid_dataloaders,disable=not accelerator.is_local_main_process,miniters=20 )):
            inputs  = data['img']
            onehotmasks = data['onehotmask']
            file_path = os.path.join(epoch_dir,data['filename'][0])

            image_encoding_results = encoder(inputs)
            results = decoder(image_encoding_results)
            for result, onehotmask,file_name in zip(results, onehotmasks, data['filename']):
                tmp_img = (result>0.).permute(1, 2, 0).detach().cpu().numpy()
                mask = (tmp_img * 255).astype(np.uint8)
                # img = 
                file_path = os.path.join(epoch_dir,file_name)
                cv2.imwrite(file_path, mask)

                target = onehotmask.to(result.device)       # shape (1, C, H, W)
                pred_prob = torch.sigmoid(result)                     # (1, C, H, W)
                pred_binary = pred_prob > 0.5                         # bool tensor

                pred_bin_np = pred_binary.cpu().numpy()    # (C, H, W) bool
                target_np = target.cpu().numpy()           # (C, H, W) float (0/1)
                # 转为 bool 以方便 &, | 运算
                target_bool = target_np > 0.5

                iou, dice, pa = compute_metrics(pred_bin_np, target_bool)
                total_iou += iou
                total_dice += dice
                total_pa += pa
                num_samples +=1


        # 6. 计算平均指标
        m_iou = total_iou / num_samples
        m_dice = total_dice / num_samples
        m_pa = total_pa / num_samples
        if best_iou< m_iou:
            best_iou = m_iou
            best_iou_epoch = epoch
            best_dice_iou = m_dice

        print(f"Epoch {epoch}: Mean IoU: {m_iou:.4f}, Mean Dice: {m_dice:.4f}, Mean PA: {m_pa:.4f} Best Epoch {best_iou_epoch:.4f}: Best IoU: {best_iou:.4f}, Best Dice IoU: {best_dice_iou:.4f}")

        with open(os.path.join(output_dir,'validation_metrics.txt'), 'a') as file:
            file.write(f'Epoch {epoch} stats:\n')  
            file.write(f"Mean IoU: {m_iou:.4f}, Mean Dice: {m_dice:.4f}, Mean PA: {m_pa:.4f}\n") 
            file.write(f"Best Epoch {best_iou_epoch:.4f}: Best IoU: {best_iou:.4f}, Best Dice IoU: {best_dice_iou:.4f}\n") 
    
def get_args_parser():
    parser = argparse.ArgumentParser("drive-sam", add_help=False)

    parser.add_argument("--output", type=str, default='result/drive_sam', 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--data_dir", type=str, default='data/sundataset', 
                        help="The path of dataset.")    
    parser.add_argument("--encoder_type", type=str, default="efficient_sam_vitt_hq", 
                        help="The type of model to loadm, type list :[efficient_sam_vitt, efficient_sam_vits]")
    parser.add_argument("--decoder_type", type=str, default="Simple", 
                        help="The type of model to loadm, type list :[Simple]")
    parser.add_argument("--encoder_checkpoint", type=str,default='weights/efficient_sam_vitt.pt', 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0,type=int)
    parser.add_argument('--lr_drop_epoch', default=4, type=int)
    parser.add_argument('--max_epoch_num', default=100, type=int)
    parser.add_argument('--fixed_size', default=1024, type=int)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=3, type=int)
    parser.add_argument('--use_argumentation',default=True,type=bool)
    parser.add_argument('--eval', default=True, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--use_patch_embed', default=True, action='store_true')
    parser.add_argument('--scale_mode', default='bilinearconv_hq', type=str,
                        help="[BDMv2, BDMv1, bilinear, nearest, bilinearconv, nearestconv]")
    parser.add_argument('--in_channels', default=3, help='The inchannel of inputs')
    parser.add_argument('--hidden_channels', default=256, help='The inchannel of inputs')
    parser.add_argument('--out_channels', default=3, help='The inchannel of inputs')
    parser.add_argument('--decoder_use_outconv', default=True, type=bool)
    parser.add_argument('--num_rank', default=128, type=int,help="the number of rank")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args_parser()
    output_dir = os.path.join(args.output,args.encoder_type, args.decoder_type, args.scale_mode)
    train_dataset = SUNDRONE_Dataset(
        datasetroot=args.data_dir,
        split='train',
        use_argumentation=args.use_argumentation
    )
    valid_dataset = SUNDRONE_Dataset(
        datasetroot=args.data_dir,
        split='val',
        use_argumentation=False
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],project_dir=output_dir)
    set_seed(args.seed)



    ### --- Step 1: Loading data---
    train_dataloaders = DataLoader(train_dataset,batch_size=args.batch_size_train,shuffle=True,num_workers=8,drop_last=True,collate_fn=custom_collate)
    if accelerator.is_main_process:
        print("--- create training dataloader ---")
        print(len(train_dataloaders), " train dataloaders created")
    valid_dataloaders = DataLoader(valid_dataset,batch_size=1,shuffle=False,num_workers=8,drop_last=True)
    if accelerator.is_main_process:
        print("--- create valid dataloader ---")
        print(len(valid_dataloaders), " valid dataloaders created")

    ### --- Step 2: Building the model---
    
    encoder = OurEncoder(args=args,in_channels=args.in_channels,out_channels=args.hidden_channels,fixed_size=args.fixed_size,scale_mode=args.scale_mode,encoder_type=args.encoder_type)
    LORA(model=encoder.feature_encoder,rank=args.num_rank)
    decoder = OurDecoder(args=args, in_channels=args.hidden_channels, out_channels=args.out_channels, fixed_size=args.fixed_size, scale_mode=args.scale_mode, decoder_mode=args.encoder_type, checkpoint=None)
   
    encoder = encoder.to(device=args.device)
    decoder = decoder.to(device=args.device)
    encoder_params = [
        {'params': encoder.parameters()},
    ]
    decoder_params = [
        {'params': decoder.parameters()}
    ]
    optimizer_encoder = optim.Adam(encoder_params, lr=1e-4*math.sqrt(args.batch_size_train), betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer_decoder = optim.Adam(decoder_params, lr=args.learning_rate*math.sqrt(args.batch_size_train), betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_decoder, args.lr_drop_epoch)
    
    encoder, optimizer_encoder, decoder, optimizer_decoder, train_dataloaders, valid_dataloaders, lr_scheduler = accelerator.prepare(
        encoder, optimizer_encoder, decoder, optimizer_decoder, train_dataloaders, valid_dataloaders, lr_scheduler
    )
    train(args, encoder, optimizer_encoder, decoder, optimizer_decoder, train_dataloaders, valid_dataloaders, lr_scheduler, accelerator)