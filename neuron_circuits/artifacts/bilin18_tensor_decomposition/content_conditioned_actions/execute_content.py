"""Raw pretrained embeddings in; exact tokenwise x0 and cache out. No model access."""
import torch
import torch.nn.functional as F

def content(weights, embeddings):
    if embeddings.dtype != torch.float32 or embeddings.shape[-1] != 1152:
        raise ValueError("Expected float32 raw token embeddings of width1152")
    x0=F.rms_norm(embeddings,(1152,))
    raw=weights['residual0']*x0+weights['embedding0']*x0
    cache=F.linear(F.rms_norm(raw,(1152,)),weights['value'])
    return x0,cache.reshape(*cache.shape[:-1],9,128)
