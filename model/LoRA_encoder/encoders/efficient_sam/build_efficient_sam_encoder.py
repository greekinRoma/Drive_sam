from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt
def build_efficient_sam_vits_encoder(checkpoint_path):
    return build_efficient_sam_vits(checkpoint_path=checkpoint_path).image_encoder
def build_efficient_sam_vitt_encoder(checkpoint_path):
    return build_efficient_sam_vitt(checkpoint_path=checkpoint_path).image_encoder

