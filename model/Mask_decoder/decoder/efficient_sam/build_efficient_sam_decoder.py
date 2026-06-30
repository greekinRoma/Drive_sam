from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt
def build_efficient_sam_vits_decoder(checkpoint_path):
    sam_model = build_efficient_sam_vits(checkpoint_path=checkpoint_path)
    return sam_model.mask_decoder, sam_model.prompt_encoder
def build_efficient_sam_vitt_decoder(checkpoint_path):
    sam_model = build_efficient_sam_vits(checkpoint_path=checkpoint_path)
    return sam_model.mask_decoder, sam_model.prompt_encoder
