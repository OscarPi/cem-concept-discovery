import torch

def cos_similarity_cubed_single(clip_feats, target_feats):
    """
    Substract mean from each vector, then raises to third power and compares cos similarity
    Does not modify any tensors in place
    Only compares first neuron to first concept etc.
    """

    clip_feats = clip_feats.float()
    clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
    target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)

    clip_feats = clip_feats**3
    target_feats = target_feats**3

    clip_feats = clip_feats/torch.norm(clip_feats, p=2, dim=0, keepdim=True)
    target_feats = target_feats/torch.norm(target_feats, p=2, dim=0, keepdim=True)

    similarities = torch.sum(target_feats*clip_feats, dim=0)
    return similarities
