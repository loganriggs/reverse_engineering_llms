"""Remove complete QK1*QK2*V-to-OV head paths on supported rule edits."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_surface_mixture_edits import predict
from eval_surface_mixture_v2_edits import build_pairs
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


def mean_l1(x):
    return float(np.abs(x).sum(-1).mean())


def summarize(pairs, p0, p1, a0, a1, vocab):
    result = {}
    for kind in sorted({pair['kind'] for pair in pairs}):
        indices = [i for i, pair in enumerate(pairs) if pair['kind'] == kind]
        native = p1[indices]-p0[indices]
        ablated = a1[indices]-a0[indices]
        oracle = np.zeros_like(native)
        for j, i in enumerate(indices):
            for token, mass in pairs[i]['q_after'].items():
                oracle[j, vocab[token]] += mass
            for token, mass in pairs[i]['q_before'].items():
                oracle[j, vocab[token]] -= mass
        removed = native-ablated
        norm2 = np.square(native).sum(-1).mean()
        result[kind] = dict(n=len(indices), native_effect_tv=mean_l1(native)/2,
                            ablated_effect_tv=mean_l1(ablated)/2,
                            removed_effect_l1=mean_l1(removed),
                            aligned_removed_fraction=float(
                                (native*removed).sum(-1).mean()/norm2),
                            native_to_oracle_l1=mean_l1(native-oracle),
                            ablated_to_oracle_l1=mean_l1(ablated-oracle))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=20261129)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    saved = json.loads((a.model_dir/'results.json').read_text())
    assert saved['config']['generator'] == 'v2_varied'
    gen = SurfaceMixtureV2Varied(seed=a.seed, **saved['generator_kwargs'])
    pairs = build_pairs(gen, a.n, a.length)
    before = [pair['original'] for pair in pairs]
    after = [pair['edited'] for pair in pairs]
    record = saved['records'][0]
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    p0 = predict(model, before, len(gen.vocab))
    p1 = predict(model, after, len(gen.vocab))
    rows = {}
    for layer in range(model.config.n_layer):
        bond = model.blocks[layer].attn.out
        head_width = model.config.n_embd//model.config.n_head
        for head in list(range(model.config.n_head))+['all']:
            def remove(_, inputs):
                value = inputs[0].clone()
                if head == 'all':
                    value.zero_()
                else:
                    value[..., head*head_width:(head+1)*head_width] = 0
                return (value,)
            hook = bond.register_forward_pre_hook(remove)
            try:
                a0 = predict(model, before, len(gen.vocab))
                a1 = predict(model, after, len(gen.vocab))
            finally:
                hook.remove()
            name = f'layer{layer}_head{head}'
            rows[name] = summarize(pairs, p0, p1, a0, a1, gen.token2id)
            print(name, 'induction', round(
                rows[name]['induction']['removed_effect_l1'], 3),
                'place', round(rows[name]['place_skip']['removed_effect_l1'], 3),
                flush=True)
    a.output.write_text(json.dumps(dict(
        config={**vars(a), 'model_dir': str(a.model_dir),
                'output': str(a.output)},
        n_pairs=len(pairs), ablations=rows,
        caveat='A complete head is removed after QK1*QK2*V and before its '
               'output projection. This does not separately identify QK1, '
               'QK2, OV, or direct residual paths; effects may interact.'),
        indent=2)+'\n')


if __name__ == '__main__':
    main()
