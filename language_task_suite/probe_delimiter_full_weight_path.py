"""Weight-only, QK×OV→MLP→readout delimiter-path probe.

With first attention zeroed, the two-block no-norm bilinear model has an
exact tokenwise table after its first MLP. For a hypothetical sequence with
one opener, zero middle positions, and one query token, this script computes
the exact closer-logit margin through second attention and the final MLP.
The calculation is checked against an ordinary model forward pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_corpus_delimiter_bilinear import _model


def _rotate(q, cos, sin):
    left, right = q.chunk(2, dim=-1)
    return q*cos + torch.cat((-right, left), dim=-1)*sin


@torch.no_grad()
def margin(model, effective_table, query_ids, distances,
           opener_ids, closer_ids, ablate_head=None,
           include_final_mlp=True):
    """Exact 2-token path margin, first attention fixed to zero output."""
    attn = model.layers[2]
    final_mlp = model.layers[3]
    assert attn.norm.__class__.__name__ == 'Identity'
    hq = effective_table[query_ids]
    ho = effective_table[opener_ids]
    b, n_head, width = len(query_ids), attn.n_head, attn.d_head
    q1 = F.linear(hq, attn.q1.weight).view(b, n_head, width)
    q2 = F.linear(hq, attn.q2.weight).view(b, n_head, width)
    kq1 = F.linear(hq, attn.k1.weight).view(b, n_head, width)
    kq2 = F.linear(hq, attn.k2.weight).view(b, n_head, width)
    ko1 = F.linear(ho, attn.k1.weight).view(b, n_head, width)
    ko2 = F.linear(ho, attn.k2.weight).view(b, n_head, width)
    cos = attn.rotary.cos_cached[0, distances]
    sin = attn.rotary.sin_cached[0, distances]
    r1 = _rotate(q1, cos, sin)
    r2 = _rotate(q2, cos, sin)
    # The key at position 0 has identity rotation. Self QK dot products
    # cancel their equal rotations and can be computed at position 0.
    key_score = (r1*ko1).sum(-1)*(r2*ko2).sum(-1)/width**2
    self_score = (q1*kq1).sum(-1)*(q2*kq2).sum(-1)/width**2
    vo = F.linear(ho, attn.v.weight).view(b, n_head, width)
    vq = F.linear(hq, attn.v.weight).view(b, n_head, width)
    z = key_score[..., None]*vo + self_score[..., None]*vq
    if ablate_head is not None:
        z[:, ablate_head] = 0
    z = z.reshape(b, n_head*width)
    residual = .5*hq + .5*F.linear(z, attn.o.weight)
    final = final_mlp(residual) if include_final_mlp else residual
    readout = (model.head.weight[closer_ids[0]]-
               model.head.weight[closer_ids[1]])
    return (final*readout).sum(-1)


@torch.no_grad()
def check_exact(model, effective_table, opener_ids, closer_ids):
    device = effective_table.device
    query_ids = torch.tensor([11, 42, 777, 15000], device=device)
    for distance in (1, 8):
        x = torch.zeros(len(query_ids), distance+1,
                        effective_table.size(-1), device=device)
        x[:, 0] = model.embed.weight[opener_ids[0]]
        x[:, distance] = model.embed.weight[query_ids]
        def zero_first_attention(module, inputs):
            return (torch.zeros_like(inputs[0]),)
        hook = model.layers[0].o.register_forward_pre_hook(
            zero_first_attention)
        try:
            for layer in model.layers:
                x = layer(x)
        finally:
            hook.remove()
        readout = (model.head.weight[closer_ids[0]]-
                   model.head.weight[closer_ids[1]])
        direct = (x[:, distance]*readout).sum(-1)
        distances = torch.full((len(query_ids),), distance,
                               dtype=torch.long, device=device)
        exact = margin(model, effective_table, query_ids, distances,
                       opener_ids[0].expand_as(query_ids), closer_ids)
        torch.testing.assert_close(exact, direct, rtol=1e-4, atol=1e-4)
    return True


@torch.no_grad()
def score_population(model, table, query_ids, distances,
                     opener_ids, closer_ids, batch_size,
                     include_final_mlp=True):
    head_count = model.layers[2].n_head
    full_values = []
    ablated_values = [[] for _ in range(head_count)]
    for start in range(0, len(query_ids), batch_size):
        query = query_ids[start:start+batch_size]
        delta = distances[start:start+batch_size]
        outputs = []
        for opener in opener_ids:
            opener_batch = opener.expand_as(query)
            outputs.append(margin(model, table, query, delta,
                                  opener_batch, closer_ids,
                                  include_final_mlp=include_final_mlp))
        full = outputs[0]-outputs[1]
        full_values.append(full.cpu())
        for head in range(head_count):
            outputs = []
            for opener in opener_ids:
                outputs.append(margin(model, table, query, delta,
                                      opener.expand_as(query), closer_ids,
                                      ablate_head=head,
                                      include_final_mlp=include_final_mlp))
            ablated_values[head].append((outputs[0]-outputs[1]).cpu())
    full = torch.cat(full_values)
    ablated = [torch.cat(values) for values in ablated_values]
    def summary(values):
        return dict(mean=values.mean().item(),
                    median=values.median().item(),
                    positive_fraction=(values > 0).float().mean().item(),
                    mean_absolute=values.abs().mean().item())
    return dict(n=len(full), full=summary(full),
                heads=[dict(head=head,
                            ablated=summary(value),
                            mean_signed_drop=(full-value).mean().item(),
                            mean_absolute_drop=(full-value).abs().mean().item())
                       for head, value in enumerate(ablated)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache-path', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_delimiter_pilot/dataset.pt'))
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    data = torch.load(args.cache_path, map_location='cpu', weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=True)
    assert checkpoint['depth'] == 2 and checkpoint['norm'] == 'none'
    device = torch.device(args.device)
    model = _model(data['vocab_size'], data['n_ctx'], 2,
                   checkpoint['d_model'], checkpoint['heads'],
                   device, 'none')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    opener_ids = torch.tensor([
        tokenizer.encode(' '+char, add_special_tokens=False).ids[0]
        for char in ('(', '[')], device=device)
    closer_ids = torch.tensor([
        tokenizer.encode(char, add_special_tokens=False).ids[0]
        for char in (')', ']')], device=device)
    with torch.no_grad():
        table = model.layers[1](.5*model.embed.weight.detach())
    check_exact(model, table, opener_ids, closer_ids)
    vocab = torch.arange(data['vocab_size'], device=device)
    distances = torch.tensor([1, 8, 32, 64], device=device)
    uniform_ids = vocab.repeat_interleave(len(distances))
    uniform_distances = distances.repeat(len(vocab))
    train = data['train']
    train_query_ids = train['x'][
        torch.arange(len(train['x'])), train['site_position']].to(device)
    train_distances = (train['site_position']-
                       train['opener_position']).to(device)
    assert train_distances.min() >= 1
    assert train_distances.max() < data['n_ctx']
    with torch.no_grad():
        uniform = score_population(
            model, table, uniform_ids, uniform_distances,
            opener_ids, closer_ids, args.batch_size)
        task_site = score_population(
            model, table, train_query_ids, train_distances,
            opener_ids, closer_ids, args.batch_size)
        ov_readout_uniform = score_population(
            model, table, uniform_ids, uniform_distances,
            opener_ids, closer_ids, args.batch_size,
            include_final_mlp=False)
        ov_readout_task_site = score_population(
            model, table, train_query_ids, train_distances,
            opener_ids, closer_ids, args.batch_size,
            include_final_mlp=False)
    result = dict(checkpoint=str(args.checkpoint),
                  exact_forward_check=True,
                  model_approximation='first attention O input fixed at zero; '
                                      'all later computations exact for a '
                                      'two-token/zero-middle sequence',
                  uniform_vocabulary=uniform,
                  training_task_site_queries=task_site,
                  ov_readout_uniform=ov_readout_uniform,
                  ov_readout_training_task_site=ov_readout_task_site,
                  caveat='Weight-only uniform vocabulary and data-aware '
                         'training-site settings are simplified contexts; '
                         'compare against causal ablation on held-out natural '
                         'documents, not as a complete proof of circuit.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('uniform head drops', [round(v['mean_signed_drop'], 3)
                                 for v in uniform['heads']], flush=True)
    print('site head drops', [round(v['mean_signed_drop'], 3)
                              for v in task_site['heads']], flush=True)


if __name__ == '__main__':
    main()
