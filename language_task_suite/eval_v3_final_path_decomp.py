"""Decompose final readout edit effects into residual and attention-bond paths."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_v3_rule_holdout_bond import capture, panel, readout
from surface_mixture_v3 import SurfaceMixtureV3


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_v3_object8')
    p.add_argument('--seed', type=int, default=29101)
    p.add_argument('--streams', type=int, default=160)
    p.add_argument('--cap', type=int, default=120)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    folder = Path(a.model_dir)
    record = json.loads((folder/'results.json').read_text())['records'][0]
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     weights_only=True, map_location='cpu'))
    model.eval()
    pairs = panel(a.seed, a.streams, a.cap, a.length)
    h0, z0, p0 = capture(model, [pair['original'] for pair in pairs])
    h1, z1, p1 = capture(model, [pair['edited'] for pair in pairs])
    p_bond = readout(model, h0, z1)
    p_residual = readout(model, h1, z0)
    assert (readout(model, h0, z0)-p0).abs().max() < 1e-5
    assert (readout(model, h1, z1)-p1).abs().max() < 1e-5
    native = p1-p0
    bond = p_bond-p0
    residual = p_residual-p0
    interaction = native-bond-residual
    oracle = torch.zeros_like(native)
    vocab = SurfaceMixtureV3(object_vocab_size=8).token2id
    for i, pair in enumerate(pairs):
        for token, mass in pair['q_after'].items():
            oracle[i, vocab[token]] += mass
        for token, mass in pair['q_before'].items():
            oracle[i, vocab[token]] -= mass
    output = {}
    for kind in sorted({pair['kind'] for pair in pairs}):
        mask = torch.tensor([pair['kind'] == kind for pair in pairs])
        l1 = lambda x: x[mask].abs().sum(-1).mean().item()
        output[kind] = dict(n=int(mask.sum()), native_effect_l1=l1(native),
                            bond_only_effect_l1=l1(bond),
                            residual_only_effect_l1=l1(residual),
                            interaction_l1=l1(interaction),
                            bond_to_native_l1=l1(bond-native),
                            residual_to_native_l1=l1(residual-native),
                            native_to_oracle_l1=l1(native-oracle),
                            bond_to_oracle_l1=l1(bond-oracle),
                            residual_to_oracle_l1=l1(residual-oracle))
    result = dict(config=vars(a), pair_counts=dict(Counter(p['kind'] for p in pairs)),
                  results=output,
                  caveat='Cross-world final readout interventions exchange the '
                         'last-block input residual h and attention pre-output '
                         'activation z separately. They exactly reconstruct the '
                         'native endpoints, but interaction terms and upstream '
                         'dependencies preclude a unique mechanistic attribution.')
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    for kind in ('induction', 'object_induction'):
        print(kind, {k: round(v, 3) for k, v in output[kind].items()
                     if k.endswith('_l1')}, flush=True)


if __name__ == '__main__':
    main()
