"""Patch individual complete attention-head paths on grammar-valid rule edits."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_surface_mixture_v2_edits import build_pairs
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


def summarize(pairs, native0, native1, patched, reverse_patched, vocab):
    rows = {}
    for kind in sorted({pair['kind'] for pair in pairs}):
        indices = np.asarray([i for i, pair in enumerate(pairs)
                              if pair['kind'] == kind])
        full = native1[indices]-native0[indices]
        oracle = np.zeros_like(full)
        for j, index in enumerate(indices):
            for token, mass in pairs[index]['q_after'].items():
                oracle[j, vocab[token]] += mass
            for token, mass in pairs[index]['q_before'].items():
                oracle[j, vocab[token]] -= mass
        denom = np.square(full).sum(-1).mean()
        by_head = {}
        for name, probs in patched.items():
            effect = probs[indices]-native0[indices]
            removed = native1[indices]-reverse_patched[name][indices]
            by_head[name] = dict(
                patch_effect_l1=float(np.abs(effect).sum(-1).mean()),
                residual_to_full_l1=float(np.abs(full-effect).sum(-1).mean()),
                aligned_full_fraction=float((full*effect).sum(-1).mean()/
                                            max(denom, 1e-30)),
                reverse_removed_l1=float(np.abs(removed).sum(-1).mean()),
                reverse_residual_to_full_l1=float(
                    np.abs(full-removed).sum(-1).mean()),
                reverse_aligned_full_fraction=float(
                    (full*removed).sum(-1).mean()/max(denom, 1e-30)),
                oracle_effect_l1_error=float(np.abs(oracle-effect).sum(-1).mean()))
        rows[kind] = dict(n=len(indices),
                          full_effect_l1=float(np.abs(full).sum(-1).mean()),
                          full_oracle_effect_l1_error=float(
                              np.abs(oracle-full).sum(-1).mean()),
                          paths=by_head)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=20261203)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    metadata = json.loads((a.model_dir/'results.json').read_text())
    assert metadata['config']['generator'] == 'v2_varied'
    gen = SurfaceMixtureV2Varied(seed=a.seed, **metadata['generator_kwargs'])
    pairs = build_pairs(gen, a.n, a.length)
    record = next(row for row in metadata['records'] if row['depth'] == 2)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    n_head = model.config.n_head
    d_head = model.config.n_embd//n_head
    original = np.zeros((len(pairs), len(gen.vocab)), np.float32)
    edited = np.zeros_like(original)
    patched = {f'layer{layer}_head{head}': np.zeros_like(original)
               for layer in range(model.config.n_layer)
               for head in [*range(n_head), 'all']}
    reverse_patched = {name: np.zeros_like(original) for name in patched}
    groups = defaultdict(list)
    for i, pair in enumerate(pairs):
        if len(pair['original']) != len(pair['edited']):
            raise ValueError('Edit changed sequence length')
        groups[len(pair['original'])].append(i)
    state = dict(mode='capture', target=None, head=None, captured={})
    hooks = []
    for layer, block in enumerate(model.blocks):
        def before_out(_, inputs, layer=layer):
            value = inputs[0]
            if state['mode'] == 'capture':
                state['captured'][layer] = value.detach().clone()
                return None
            if state['target'] != layer:
                return None
            source = state['source'][layer]
            head = state['head']
            if head == 'all':
                return (source,)
            value = value.clone()
            sl = slice(head*d_head, (head+1)*d_head)
            value[..., sl] = source[..., sl]
            return (value,)
        hooks.append(block.attn.out.register_forward_pre_hook(before_out))
    self_patch_max_abs = 0.
    try:
        with torch.no_grad():
            for indices in groups.values():
                for start in range(0, len(indices), 64):
                    batch = indices[start:start+64]
                    x0 = torch.tensor([pairs[i]['original'] for i in batch],
                                      dtype=torch.long)
                    x1 = torch.tensor([pairs[i]['edited'] for i in batch],
                                      dtype=torch.long)
                    state['mode'] = 'capture'
                    logits0 = model(x0)[0][:, -1].float()
                    own = {k: v.clone() for k, v in state['captured'].items()}
                    state.update(mode='patch', source=own, target=0,
                                 head='all')
                    self_logits = model(x0)[0][:, -1].float()
                    self_patch_max_abs = max(self_patch_max_abs,
                        float((self_logits-logits0).abs().max()))
                    state.update(mode='capture', target=None)
                    logits1 = model(x1)[0][:, -1].float()
                    other = {k: v.clone() for k, v in state['captured'].items()}
                    state.update(mode='patch', source=other, target=0,
                                 head='all')
                    self_logits1 = model(x1)[0][:, -1].float()
                    self_patch_max_abs = max(self_patch_max_abs,
                        float((self_logits1-logits1).abs().max()))
                    original[batch] = F.softmax(logits0, -1).numpy()
                    edited[batch] = F.softmax(logits1, -1).numpy()
                    state.update(mode='patch', source=other)
                    for layer in range(model.config.n_layer):
                        for head in [*range(n_head), 'all']:
                            state.update(target=layer, head=head)
                            name = f'layer{layer}_head{head}'
                            patched[name][batch] = F.softmax(
                                model(x0)[0][:, -1].float(), -1).numpy()
                    state.update(source=own)
                    for layer in range(model.config.n_layer):
                        for head in [*range(n_head), 'all']:
                            state.update(target=layer, head=head)
                            name = f'layer{layer}_head{head}'
                            reverse_patched[name][batch] = F.softmax(
                                model(x1)[0][:, -1].float(), -1).numpy()
    finally:
        for hook in hooks:
            hook.remove()
    if self_patch_max_abs > 1e-5:
        raise AssertionError(f'Self patch changed logits: {self_patch_max_abs}')
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  n_pairs=len(pairs), self_patch_max_abs=self_patch_max_abs,
                  results=summarize(pairs, original, edited, patched,
                                    reverse_patched, gen.token2id),
                  caveat='Patches pre-output-projection head values at all '
                         'positions in both edit directions. This is '
                         'a causal path intervention, but patches at different '
                         'layers/heads interact and do not add to a unique '
                         'decomposition. The direct residual stays at the '
                         'target run.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('pairs', len(pairs), 'self check', self_patch_max_abs)
    for kind in ('lexical', 'induction', 'place_skip', 'cue'):
        if kind not in result['results']:
            continue
        print(kind, {name: round(row['aligned_full_fraction'], 3)
                     for name, row in result['results'][kind]['paths'].items()},
              flush=True)


if __name__ == '__main__':
    main()
