"""Matched-shell causal softmax-attention control for bilinear toy LMs.

Uses the same BilinearGPT embedding, blocks, output, and Q1/K1/V projections,
but replaces each block's product-of-two-QK attention with ordinary masked
softmax attention. Q2/K2 are stored but inactive, so active parameter count is
smaller; this is a capability control, not a parameter-matched comparison.
"""

import math
import types

import torch
import torch.nn.functional as F


def _softmax_forward(self, x, return_max_logit=False):
    from model import apply_rotary_emb

    batch, length, width = x.shape
    head_dim = width // self.n_head
    q = self.q1(x).view(batch, length, self.n_head, head_dim)
    k = self.k1(x).view(batch, length, self.n_head, head_dim)
    v = self.v(x).view(batch, length, self.n_head, head_dim)
    cos, sin = self.rotary(q)
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    if self.qk_norm:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(head_dim)
    mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, -torch.inf)
    pattern = F.softmax(scores, dim=-1)
    y = torch.einsum('bhqk,bkhd->bqhd', pattern, v)
    out = self.out(y.reshape(batch, length, width))
    return (out, float(scores.masked_fill(~mask, 0.).abs().max())) if return_max_logit else out


def configure_attention(model, variant):
    if variant == 'bilinear':
        return model
    if variant != 'softmax':
        raise ValueError(variant)
    for block in model.blocks:
        block.attn.forward = types.MethodType(_softmax_forward, block.attn)
    return model
