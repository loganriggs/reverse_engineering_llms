"""M74 native-write directional derivatives and downstream Fisher metric."""
import torch
from inspect_model import trace


@torch.no_grad()
def directional_logits(model, ids, residual):
    """Forward-mode AD through the unchanged traced model, including softcap.

    The primal patch is zero; the tangent is the fixed product-write residual.
    Native bias, embedding skip and first-layer value bus remain unchanged.
    """
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.zeros_like(residual), residual)
        out, _ = trace(model, ids, replace=lambda j, z, w: w+dual if j == 1 else w)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(out)
        assert tangent is not None and torch.isfinite(tangent).all()
        return primal.clone(), tangent.clone()


def fisher_energy(logits, tangent):
    probability = logits.double().softmax(-1)
    tangent = tangent.double()
    centered = tangent-(probability*tangent).sum(-1, keepdim=True)
    return (probability*centered.square()).sum(-1)


@torch.no_grad()
def perturbed_logits(model, ids, residual, alpha):
    out, _ = trace(model, ids, replace=lambda j, z, w: w+alpha*residual if j == 1 else w)
    return out
