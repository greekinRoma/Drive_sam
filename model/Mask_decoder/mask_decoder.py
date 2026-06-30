from torch import nn
import torch
from .decoder import build_decoder
from ..Scalar.Scalar_reconstruction import Scalar_Reconstruction, Upsample4xSequential
from .upsample.upsample_hq import Upsample4xSequential_hq
from .resblock import ResidualBlock
class OurDecoder(nn.Module):
    def __init__(self,args, in_channels, out_channels, fixed_size, scale_mode, decoder_mode, checkpoint=None):
        super().__init__()
        self.scale_mode = scale_mode
        print(self.scale_mode)
        if self.scale_mode in ['nearest', 'bilinear', 'nearestconv', 'bilinearconv']: 
            self.featuresdecoder, prompt_encoder = build_decoder(decoder_mode=decoder_mode,checkpoint=checkpoint,in_channel=in_channels,out_channel=in_channels)
            with torch.no_grad():
                box_corners = torch.tensor([[[0.0, 0.0], [1024, 1024]]], device=args.device)
                points = box_corners.unsqueeze(0).expand(1, 1, -1, -1)
                point_labels = torch.full((1, 1, 2), 2, dtype=torch.int, device=args.device)
                point_labels[..., 1] = 3
                prompt_encoder = prompt_encoder.cuda()
                sparse_embeddings = prompt_encoder(points.reshape(1, 2, 2),point_labels.reshape(1 , 2))
                self.sparse_embeddings = sparse_embeddings.view(1,1,sparse_embeddings.shape[1],sparse_embeddings.shape[2])
                self.image_pe=prompt_encoder.get_dense_pe()
                self.scalar_reconstruction = Scalar_Reconstruction(in_channels=in_channels, out_channels=out_channels, fixed_size=fixed_size, mode=scale_mode,use_outconv=args.decoder_use_outconv)
                self.upsample = Upsample4xSequential(in_channels=in_channels)
        elif self.scale_mode in ["nearestconv_hq", "bilinearconv_hq"]:
            self.compress_vit_feat = ResidualBlock(in_channels=192,out_channels=in_channels)
            self.scalar_reconstruction = Scalar_Reconstruction(in_channels=in_channels, out_channels=out_channels, fixed_size=fixed_size, mode=scale_mode,use_outconv=args.decoder_use_outconv)
            self.fuse_layer = nn.Sequential(*[nn.Conv2d(in_channels=in_channels+in_channels,out_channels=in_channels,kernel_size=1,stride=1),
                                            nn.BatchNorm2d(in_channels),
                                            nn.GELU(),
                                            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1),
                                            nn.BatchNorm2d(in_channels),
                                            nn.GELU()])
            self.upsample = Upsample4xSequential(in_channels=in_channels)
        
    def forward(self, encoding_results):
        outs = []
        if self.scale_mode in ['nearest', 'bilinear', 'nearestconv', 'bilinearconv']: 
            image_embeddings = encoding_results['image_embedding']
            origin_sizes = encoding_results['original_sizes']
            image_embeddings = self.upsample(image_embeddings)
            for image_embeding, origin_size in zip(image_embeddings, origin_sizes):
                out = self.scalar_reconstruction(x=image_embeding, original_size=origin_size)
                outs.append(out)
        elif self.scale_mode in ['nearestconv_hq', 'bilinearconv_hq']: 
            image_embeddings = encoding_results['image_embedding']
            feature_embeddings = encoding_results['feature_embedding']
            vit_feature = feature_embeddings[0]
            # print(len(feature_embeddings))
            # feature_embedding = torch.stack(feature_embeddings,dim=0)
            # vit_feature = torch.mean(feature_embedding,dim=0)
            b, hw, c = vit_feature.shape
            vit_features = self.compress_vit_feat(vit_feature.reshape(b, 64, 64, c).permute(0, 3, 1, 2))
            image_embeddings = self.fuse_layer(torch.concat([vit_features, image_embeddings],dim=1))
            image_embeddings = self.upsample(image_embeddings)
            origin_sizes = encoding_results['original_sizes']
            for image_embeding, origin_size in zip(image_embeddings, origin_sizes):
                out = self.scalar_reconstruction(x=image_embeding, original_size=origin_size)
                outs.append(out)
        else:
            raise f"No mode of scaling is {self.scale_mode}"
        outs = [out[0] for out in outs]
        return outs
