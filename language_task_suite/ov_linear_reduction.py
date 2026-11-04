"""Generic OV gauge and rank reduction for 64-wide bilinear attention heads."""

from __future__ import annotations

import torch


def attentions(model):
    return [layer for layer in model.layers if hasattr(layer, 'q1')]


@torch.no_grad()
def activation_grams(model, rows, device, batch_size=4):
    layers = attentions(model)
    grams = [torch.zeros((layer.v.in_features, layer.v.in_features),
                         dtype=torch.float64) for layer in layers]
    counts = [0]*len(layers)
    for start in range(0, len(rows['x']), batch_size):
        x = rows['x'][start:start+batch_size].to(device)
        mask = rows['y'][start:start+batch_size].to(device) != -100
        hooks = []
        for index, layer in enumerate(layers):
            def capture(module, inputs, index=index):
                values = inputs[0][mask].detach().to('cpu', torch.float64)
                grams[index].add_(values.T @ values)
                counts[index] += len(values)
            hooks.append(layer.v.register_forward_pre_hook(capture))
        model(x)
        for hook in hooks:
            hook.remove()
    assert all(count > 0 for count in counts)
    return [gram/count for gram, count in zip(grams, counts)], counts


def _sqrt_gram(gram, ridge_fraction):
    ridge = ridge_fraction*torch.trace(gram)/len(gram)
    values, basis = torch.linalg.eigh(gram+ridge*torch.eye(len(gram)))
    values = values.clamp_min(1e-12)
    return ((basis*values.sqrt()) @ basis.T,
            (basis*values.rsqrt()) @ basis.T)


def _reduce(matrix, rank, metric):
    if metric is None:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        return (u[:, :rank]*s[:rank]) @ vh[:rank]
    sqrt, inverse = metric
    u, s, vh = torch.linalg.svd(matrix @ sqrt, full_matrices=False)
    return ((u[:, :rank]*s[:rank]) @ vh[:rank]) @ inverse


@torch.no_grad()
def replace_ov(model, grams, rank, method, ridge_fraction=.001):
    assert method in ('weight', 'data')
    errors = []
    for layer_index, layer in enumerate(attentions(model)):
        metric = (_sqrt_gram(grams[layer_index], ridge_fraction)
                  if method == 'data' else None)
        d_head = layer.d_head
        assert 1 <= rank <= d_head
        for head in range(layer.n_head):
            sl = slice(head*d_head, (head+1)*d_head)
            o = layer.o.weight[:, sl].detach().cpu().double()
            v = layer.v.weight[sl, :].detach().cpu().double()
            original = o @ v
            reduced = _reduce(original, rank, metric)
            u, s, vh = torch.linalg.svd(reduced, full_matrices=False)
            new_o = torch.zeros_like(o)
            new_v = torch.zeros_like(v)
            new_o[:, :rank] = u[:, :rank]*s[:rank].sqrt()
            new_v[:rank] = s[:rank, None].sqrt()*vh[:rank]
            layer.o.weight[:, sl].copy_(new_o.to(
                layer.o.weight.device, layer.o.weight.dtype))
            layer.v.weight[sl, :].copy_(new_v.to(
                layer.v.weight.device, layer.v.weight.dtype))
            delta = reduced-original
            gram = grams[layer_index]
            weighted = torch.trace(delta @ gram @ delta.T)
            base = torch.trace(original @ gram @ original.T)
            errors.append({
                'layer': layer_index, 'head': head,
                'weight_rel_error': float(torch.linalg.norm(delta)/
                                          torch.linalg.norm(original)),
                'input_weighted_rel_error': float(
                    (weighted/base).clamp_min(0).sqrt())})
    return errors
