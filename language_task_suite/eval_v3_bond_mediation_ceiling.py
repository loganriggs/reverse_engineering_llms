"""Find attention bonds that mediate valid verb/object edits at full width."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from eval_surface_edit_bases import infer
from eval_v3_rule_holdout_bond import panel
from surface_mixture_v3 import SurfaceMixtureV3


def grouped_scores(pairs, base, donor, patched, vocab):
    out = {}
    for kind in sorted({pair['kind'] for pair in pairs}):
        indices = [i for i, pair in enumerate(pairs) if pair['kind'] == kind]
        model_effect = donor[indices]-base[indices]
        patch_effect = patched[indices]-base[indices]
        oracle_effect = np.zeros_like(model_effect)
        for j, i in enumerate(indices):
            for token, mass in pairs[i]['q_after'].items():
                oracle_effect[j, vocab[token]] += mass
            for token, mass in pairs[i]['q_before'].items():
                oracle_effect[j, vocab[token]] -= mass
        l1 = lambda x: float(np.abs(x).sum(-1).mean())
        out[kind] = dict(n=len(indices),
                         patch_to_native_l1=l1(patch_effect-model_effect),
                         patch_to_oracle_l1=l1(patch_effect-oracle_effect),
                         native_to_oracle_l1=l1(model_effect-oracle_effect),
                         native_effect_l1=l1(model_effect),
                         patch_effect_l1=l1(patch_effect))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dir', default='language_task_suite/surface_mixture_v3_object8')
    parser.add_argument('--seed', type=int, default=29001)
    parser.add_argument('--streams', type=int, default=80)
    parser.add_argument('--cap', type=int, default=60)
    parser.add_argument('--length', type=int, default=72)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    folder = Path(args.model_dir)
    record = json.loads((folder/'results.json').read_text())['records'][0]
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     weights_only=True, map_location='cpu'))
    model.eval()
    pairs = panel(args.seed, args.streams, args.cap, args.length)
    before = [pair['original'] for pair in pairs]
    after = [pair['edited'] for pair in pairs]
    vocab = SurfaceMixtureV3(object_vocab_size=8).token2id
    base, _ = infer(model, model.blocks[-1].attn.out, before)
    donor, _ = infer(model, model.blocks[-1].attn.out, after)
    sites = {}
    for layer in range(model.config.n_layer):
        bond = model.blocks[layer].attn.out
        for scope in ('last', 'all'):
            full = scope == 'all'
            _, donor_acts = infer(model, bond, after, capture_full=full)
            patch, _ = infer(model, bond, before, donor_acts,
                             torch.eye(model.config.n_embd),
                             model.config.n_embd, capture_full=full)
            name = f'layer{layer}_{scope}'
            sites[name] = grouped_scores(pairs, base, donor, patch, vocab)
            print(name, {kind: round(sites[name][kind]['patch_to_native_l1'], 3)
                         for kind in ('induction','object_induction')}, flush=True)
    output = dict(config=vars(args), pair_counts=dict(Counter(p['kind'] for p in pairs)),
                  sites=sites,
                  caveat='Full-width interventions at pre-output attention bonds '
                         'give local mediation ceilings. The model was trained on all '
                         'rules. This site-selection panel must not be reused as an '
                         'untouched test for fitted bases or path groups.')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
