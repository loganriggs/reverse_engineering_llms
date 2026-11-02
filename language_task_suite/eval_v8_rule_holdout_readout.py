"""Compare final-readout bases under whole-task or edit-family holdouts.

The frozen v8 teacher was trained on every rule. Source fit withholds the
whole section task from featurizer fit and development. Section-key fit
withholds color edits instead. Neither is a training-task holdout for the
language model or new-task discovery.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from eval_surface_mixture_v6_sources import build as source_build
from eval_surface_mixture_v7_scope import build as scope_build
from eval_v3_rule_holdout_bond import remove_duplicates, orthogonal_columns
from eval_v3_two_path_holdout import optimize, pca, prepare, score
from surface_mixture_v8 import SurfaceMixtureV8


def generator(seed, kwargs):
    return SurfaceMixtureV8(seed=seed, **kwargs)


def source_panel(seed, kwargs, streams, cap):
    _, pairs, attempted = source_build(generator(seed, kwargs), streams, 256)
    groups = {}
    for pair in pairs:
        groups.setdefault(pair['kind'], []).append(pair)
    rng = np.random.default_rng(seed + 7721)
    selected = []
    for kind in sorted(groups):
        group = groups[kind]
        indices = rng.choice(len(group), size=min(cap, len(group)), replace=False)
        selected.extend(group[int(i)] for i in indices)
    return selected, attempted


def scope_panel(seed, kwargs, streams, cap):
    _, pairs, attempted = scope_build(generator(seed, kwargs), streams, 256, cap)
    return pairs, attempted


@torch.no_grad()
def blind_states(model, seed, kwargs, streams, cap):
    ids, _, _ = generator(seed, kwargs).sample_batch(streams, 256)
    states = []
    for start in range(0, streams, 8):
        captured = []
        hook = model.final_norm.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach()))
        model(torch.tensor(ids[start:start + 8], dtype=torch.long))
        hook.remove()
        states.append(captured[0].reshape(-1, model.config.n_embd))
    flat = torch.cat(states)
    rng = np.random.default_rng(seed + 31)
    indices = rng.choice(len(flat), size=min(cap, len(flat)), replace=False)
    return flat[indices]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--fit-family', choices=('source', 'section_key'), default='source',
                   help='Source gives a whole-section-family featurizer holdout; '
                        'section_key withholds color-edit families only.')
    p.add_argument('--fit-seed', type=int, default=44001)
    p.add_argument('--dev-seed', type=int, default=44002)
    p.add_argument('--test-seed', type=int, default=45001)
    p.add_argument('--fit-streams', type=int, default=600)
    p.add_argument('--dev-streams', type=int, default=300)
    p.add_argument('--test-streams', type=int, default=600)
    p.add_argument('--source-cap', type=int, default=80)
    p.add_argument('--scope-cap', type=int, default=120)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--lr', type=float, default=.03)
    p.add_argument('--blind-streams', type=int, default=32)
    p.add_argument('--blind-positions', type=int, default=2048)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    folder = Path(a.model_dir)
    saved = json.loads((folder / 'results.json').read_text())
    assert saved['config']['generator'] == 'v8'
    kwargs = saved['generator_kwargs']
    record = saved['records'][0]
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(folder / f"model_depth{record['depth']}.pt",
                                     weights_only=True, map_location='cpu'))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vocab = generator(0, kwargs).token2id

    if a.fit_family == 'source':
        raw_fit, attempted_fit = source_panel(a.fit_seed, kwargs, a.fit_streams, a.source_cap)
        raw_dev, attempted_dev = source_panel(a.dev_seed, kwargs, a.dev_streams, a.source_cap)
    else:
        fit_all, attempted_fit = scope_panel(a.fit_seed, kwargs, a.fit_streams, a.scope_cap)
        dev_all, attempted_dev = scope_panel(a.dev_seed, kwargs, a.dev_streams, a.scope_cap)
        raw_fit = [pair for pair in fit_all if pair['kind'] == 'section_key']
        raw_dev = [pair for pair in dev_all if pair['kind'] == 'section_key']
    raw_source_test, attempted_source_test = source_panel(
        a.test_seed, kwargs, a.test_streams, a.source_cap)
    raw_scope_test, attempted_scope_test = scope_panel(
        a.test_seed + 1, kwargs, a.test_streams, a.scope_cap)
    used = set()
    fit_pairs, fit_duplicates = remove_duplicates(raw_fit, used)
    dev_pairs, dev_duplicates = remove_duplicates(raw_dev, used)
    source_test_pairs, source_duplicates = remove_duplicates(raw_source_test, used)
    scope_test_pairs, scope_duplicates = remove_duplicates(raw_scope_test, used)
    assert fit_pairs and dev_pairs and source_test_pairs and scope_test_pairs
    if a.fit_family == 'source':
        assert not any(pair['kind'] in ('section_key', 'relevant_color',
                                       'irrelevant_color') for pair in fit_pairs + dev_pairs)
    else:
        assert all(pair['kind'] == 'section_key' for pair in fit_pairs + dev_pairs)
    assert set(pair['kind'] for pair in scope_test_pairs) == {
        'section_key', 'relevant_color', 'irrelevant_color'}

    fit = prepare(model, fit_pairs, vocab)
    dev = prepare(model, dev_pairs, vocab)
    source_test = prepare(model, source_test_pairs, vocab)
    scope_test = prepare(model, scope_test_pairs, vocab)
    states = blind_states(model, a.fit_seed + 500, kwargs,
                          a.blind_streams, a.blind_positions)
    d = model.config.n_embd
    assert 0 < a.rank <= d
    torch.manual_seed(a.fit_seed)
    _, _, vh = torch.linalg.svd(model.lm_head.weight.double(), full_matrices=False)
    bases = {
        'zero': ('combined', torch.empty(d, 0)),
        'full': ('combined', torch.eye(d)),
        'random': ('combined', orthogonal_columns(torch.randn(d, a.rank))),
        'weight_svd': ('combined', vh[:a.rank].T.float().contiguous()),
        'blind_stream_pca': ('combined', pca(states, a.rank)),
        'fit_delta_pca': ('combined', pca(fit['dr'], a.rank)),
    }
    learned, optimization = optimize(
        model, fit, dev, bases['fit_delta_pca'], 'native', a.steps, a.lr)
    bases['fit_causal'] = learned
    rows = {name: {'source_test': score(model, source_test, basis),
                   'section_test': score(model, scope_test, basis)}
            for name, basis in bases.items()}
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    basis_path = path.with_suffix('.pt')
    torch.save(bases, basis_path)
    result = {
        'config': vars(a), 'basis_checkpoint': str(basis_path),
        'fit_counts': dict(Counter(x['kind'] for x in fit_pairs)),
        'dev_counts': dict(Counter(x['kind'] for x in dev_pairs)),
        'seen_test_counts': dict(Counter(x['kind'] for x in source_test_pairs)),
        'heldout_test_counts': dict(Counter(x['kind'] for x in scope_test_pairs)),
        'attempted': {k: dict(v) for k, v in dict(
            fit=attempted_fit, dev=attempted_dev,
            seen_test=attempted_source_test,
            heldout_test=attempted_scope_test).items()},
        'duplicates_removed': dict(fit=fit_duplicates, dev=dev_duplicates,
                                   seen_test=source_duplicates,
                                   heldout_test=scope_duplicates),
        'max_endpoint_replay_error': max(
            fit['replay_error'], dev['replay_error'],
            source_test['replay_error'], scope_test['replay_error']),
        'blind_positions_used': len(states),
        'optimization': optimization, 'rows': rows,
        'caveat': 'The frozen teacher was trained on source and section rules. '
                  'Source fit withholds the whole section edit family; section_key '
                  'fit withholds relevant/irrelevant color edits only. '
                  'These are local final-readout interventions, not a whole-model '
                  'compression or discovery of an unseen task.'}
    path.write_text(json.dumps(result, indent=2) + '\n')
    for name, groups in rows.items():
        print(name, {kind: round(group['sufficient_to_native_l1'], 4)
                     for kind, group in groups['section_test'].items()}, flush=True)


if __name__ == '__main__':
    main()
