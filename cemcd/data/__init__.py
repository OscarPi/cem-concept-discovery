def get_latent_representation_size(foundation_model):
    if foundation_model == "dinov2_vitg14":
        return 1536
    elif foundation_model == "clip_vitl14":
        return 768
    else:
        raise ValueError(f"Unrecognised foundation model: {foundation_model}.")
