"""Conditional lexical cached-value write. Raw token embeddings are explicit inputs.
No model, token table, attention/QK, hidden states or suffix is accessed here.
"""
import torch
import torch.nn.functional as F

def cache_value(weights, embeddings):
    if embeddings.shape[-1] != 1152 or embeddings.dtype != torch.float32:
        raise ValueError('expected raw float32 embeddings with final dimension1152')
    x0=F.rms_norm(embeddings,(1152,))
    x=weights['residual0']*x0+weights['embedding0']*x0
    return F.linear(F.rms_norm(x,(1152,)),weights['value0_head0'])

def delta_writes(weights, recipient_embeddings, donor_embeddings):
    if recipient_embeddings.shape != donor_embeddings.shape:
        raise ValueError('recipient and donor shapes differ')
    dv=cache_value(weights,donor_embeddings)-cache_value(weights,recipient_embeddings)
    payload=F.linear(dv,weights['output13_head0'])*weights['cache_mix13']
    return payload[...,None,:]*weights['routing_mean'][...,None]
