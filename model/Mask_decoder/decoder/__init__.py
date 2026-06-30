import torch
from torch import nn
from model.Mask_decoder.decoder.efficient_sam.build_efficient_sam_decoder import build_efficient_sam_vits_decoder, build_efficient_sam_vitt_decoder
def build_decoder(decoder_mode,checkpoint=None,in_channel=1,out_channel=3):
    if decoder_mode == "Simple":
        return nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1)
    elif decoder_mode == "efficient_sam_vitt":
        return build_efficient_sam_vits_decoder(checkpoint_path=checkpoint)
    elif decoder_mode == "efficient_sam_vits":
        return build_efficient_sam_vitt_decoder(checkpoint_path=checkpoint)
    else:
        raise