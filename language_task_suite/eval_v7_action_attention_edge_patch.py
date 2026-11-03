"""Patch the changed adjective's attention edge on eligible V7 action edits.

This is a local, data-conditioned causal path diagnostic, not a weight-only
decomposition. It tests whether the learned action response travels through
specific QK scores and OV values before a weight-only method is trusted.
"""

import argparse
import json
from pathlib import Path

import torch

from eval_document_mixture_v7_action_causal import collect
from train_document_mixture_v5 import DeepModel, SPECS


def load_model(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    cfg = saved['model_kwargs']
    model = DeepModel(len(saved['vocabulary']), cfg['d_model'], cfg['n_head'],
                      SPECS[f"block{saved['depth']}"], saved['n_ctx'],
                      norm=cfg['norm'], residual=cfg['residual'],
                      attention=cfg['attention']).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved['vocabulary'], saved['n_ctx']


def batch(cases, vocab, n_ctx, device):
    original = torch.full((len(cases), n_ctx), vocab['<PAD>'],
                          dtype=torch.long, device=device)
    edited = original.clone()
    q, k, targets = [], [], []
    for i, (before, after, target) in enumerate(cases):
        assert len(before) == len(after) <= n_ctx
        changed = [j for j, (x, y) in enumerate(zip(before, after)) if x != y]
        assert len(changed) == 1 and changed[0] < len(before)-1
        original[i, :len(before)] = torch.tensor(
            [vocab[token] for token in before], device=device)
        edited[i, :len(after)] = torch.tensor(
            [vocab[token] for token in after], device=device)
        q.append(len(before)-1)
        k.append(changed[0])
        targets.append(vocab[target])
    return (original, edited,
            torch.tensor(q, device=device),
            torch.tensor(k, device=device),
            torch.tensor(targets, device=device))


def probability(model, tokens, q, targets):
    logits = model(tokens)[torch.arange(len(tokens), device=tokens.device), q]
    return logits.float().softmax(-1).gather(1, targets[:, None]).flatten()


def source_activations(model, tokens, layer_index):
    x = model.embed(tokens)
    for layer in model.layers[:layer_index]:
        x = layer(x)
    layer = model.layers[layer_index]
    return layer.pattern(x), layer.v(layer.norm(x))


def patched_probability(model, tokens, q, k, targets, layer_index,
                        head, mode, source_pattern, source_v):
    layer = model.layers[layer_index]
    original_pattern = layer.pattern
    handles = []
    rows = torch.arange(len(tokens), device=tokens.device)
    if mode.startswith('preceding_'):
        assert bool((k > 0).all())
        k = k-1
        mode = mode.removeprefix('preceding_')
    try:
        if mode in ('score_edge', 'score_row', 'score_and_value'):
            def pattern_patch(x):
                p = original_pattern(x).clone()
                if mode == 'score_row':
                    p[rows, head, q, :] = source_pattern[rows, head, q, :]
                else:
                    p[rows, head, q, k] = source_pattern[rows, head, q, k]
                return p
            layer.pattern = pattern_patch
        if mode in ('value_source', 'score_and_value'):
            lo, hi = head*layer.d_head, (head+1)*layer.d_head
            def value_patch(_module, _inputs, output):
                out = output.clone()
                out[rows, k, lo:hi] = source_v[rows, k, lo:hi]
                return out
            handles.append(layer.v.register_forward_hook(value_patch))
        return probability(model, tokens, q, targets)
    finally:
        layer.pattern = original_pattern
        for handle in handles:
            handle.remove()


@torch.no_grad()
def run(model, cases, vocab, n_ctx, device, batch_size):
    attention_layers = [i for i, layer in enumerate(model.layers)
                        if hasattr(layer, 'pattern')]
    modes = ('score_edge', 'score_row', 'value_source', 'score_and_value',
             'preceding_score_edge', 'preceding_value_source',
             'preceding_score_and_value')
    results = {(i, h, mode): [] for i in attention_layers
               for h in range(model.layers[i].n_head) for mode in modes}
    original_all, edited_all = [], []
    for start in range(0, len(cases), batch_size):
        part = cases[start:start+batch_size]
        before, after, q, k, target = batch(part, vocab, n_ctx, device)
        p0 = probability(model, before, q, target)
        p1 = probability(model, after, q, target)
        original_all.extend(p0.cpu().tolist())
        edited_all.extend(p1.cpu().tolist())
        for layer_index in attention_layers:
            source_p, source_v = source_activations(
                model, before, layer_index)
            if start == 0:
                replay = patched_probability(
                    model, before, q, k, target, layer_index, 0,
                    'score_and_value', source_p, source_v)
                torch.testing.assert_close(replay, p0, atol=1e-6, rtol=1e-6)
            for head in range(model.layers[layer_index].n_head):
                for mode in modes:
                    pp = patched_probability(model, after, q, k, target,
                                             layer_index, head, mode,
                                             source_p, source_v)
                    results[layer_index, head, mode].extend(pp.cpu().tolist())
    p0 = torch.tensor(original_all)
    p1 = torch.tensor(edited_all)
    response = p1-p0
    assert response.mean() > .5, 'The tested rule must be behaviorally learned'
    rows = []
    for (layer_index, head, mode), values in results.items():
        patched = torch.tensor(values)
        reduction = p1-patched
        rows.append(dict(layer=layer_index, head=head, patch=mode,
                         mean_patched_target_probability=patched.mean().item(),
                         mean_response_removed=reduction.mean().item(),
                         fraction_of_response_removed=(reduction.mean()/
                                                       response.mean()).item(),
                         positive_case_fraction=(reduction > 0).float().mean().item()))
    return dict(n=len(cases), native_target_probability_before=p0.mean().item(),
                native_target_probability_after=p1.mean().item(),
                native_response=response.mean().item(), rows=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--seed', type=int, default=35051)
    p.add_argument('--streams', type=int, default=400)
    p.add_argument('--max-cases', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    model, vocab, n_ctx = load_model(a.checkpoint, a.device)
    cases, available = collect(a.seed, a.streams, 192, a.max_cases)
    assert len(cases) == a.max_cases
    result = run(model, cases, vocab, n_ctx, a.device, a.batch_size)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
        'checkpoint': str(a.checkpoint), 'output': str(a.output)},
        available_supported_edits=available, result=result,
        caveat='Patching old score/value activations on the edited input '
               'is a local causal diagnostic. It is not an exact circuit '
               'decomposition: other paths and later nonlinearities remain.'),
        indent=2)+'\n')
    print('native response', result['native_response'])
    for row in sorted(result['rows'],
                      key=lambda r: -r['mean_response_removed'])[:12]:
        print(row['layer'], row['head'], row['patch'],
              round(row['mean_response_removed'], 4))


if __name__ == '__main__':
    main()
