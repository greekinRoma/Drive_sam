from model.LoRA_encoder.encoders.efficient_sam.build_efficient_sam_encoder import build_efficient_sam_vits_encoder, build_efficient_sam_vitt_encoder
from model.LoRA_encoder.encoders.efficient_sam_hq.build_efficient_sam_encoder import build_efficient_sam_vits_encoder_hq, build_efficient_sam_vitt_encoder_hq
from torch import nn
import torch
from ...Scalar.Scalar_decomposition import Scalar_Decomposition
def build_encoder(encoder_type="efficient_sam_vitt",check_point_path="weights/efficient_sam_vitt.pt"):
    if encoder_type == "efficient_sam_vitt":
        return build_efficient_sam_vitt_encoder(checkpoint_path=check_point_path)
    elif encoder_type == "efficient_sam_vits":
        return build_efficient_sam_vits_encoder(checkpoint_path=check_point_path)
    elif encoder_type == "efficient_sam_vitt_hq":
        return build_efficient_sam_vitt_encoder_hq(checkpoint_path=check_point_path.strip("_hq"))
    elif encoder_type == "efficient_sam_vits_hq":
        return build_efficient_sam_vits_encoder_hq(checkpoint_path=check_point_path.strip("_hq"))
    else:
        raise "The encoder is not included!!!!"

class OurEncoder(nn.Module):
    def __init__(self,args, in_channels, out_channels, fixed_size, scale_mode='bilinear',  encoder_type="efficient_sam_vitt",check_point_path="weights/efficient_sam_vitt.pt"):
        super().__init__()
        self.args = args
        self.encoder_type = encoder_type
        self.feature_encoder = build_encoder(encoder_type=encoder_type,check_point_path=check_point_path)
        self.scalar_decomposition = Scalar_Decomposition(in_channels=in_channels, out_channels=out_channels, fixed_size=fixed_size, mode=scale_mode)
    def forward(self, imgs):
        batched_inputs = []
        original_sizes = []
        for b_i in range(len(imgs)):
            input_image = torch.as_tensor(imgs[b_i]/255., device=self.args.device)
            input_image = self.scalar_decomposition(input_image)
            batched_inputs.append(input_image)
            original_sizes.append(imgs[b_i].shape[-2:])
        input_images = torch.concat(batched_inputs, dim=0)
        if self.encoder_type in ["efficient_sam_vitt" , "efficient_sam_vits"]:
            image_embeddings = self.feature_encoder(input_images)
        else:
            image_embeddings, feature_embeddings = self.feature_encoder(input_images)
        image_encoding_result = dict()
        image_encoding_result['image_embedding'] = image_embeddings
        image_encoding_result['feature_embedding'] = feature_embeddings
        image_encoding_result['original_sizes'] = original_sizes
        return image_encoding_result
