"""Held-out sufficiency and necessity audit for teacher-selected source gates."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from distill_sparse_depth_vs_factor import build_students
from eval_surface_mixture_v2_edits import build_pairs
from surface_mixture_v2 import VERBS
from surface_mixture_v2_varied import SurfaceMixtureV2Varied
from train_varied_sparse_path_transfer import source_components, teacher_indices


def branch_probabilities(model, ids, indices):
    """Native, retained-only, and removed-source probability distributions."""
    attn = model.blocks[1].attn
    contributions, residual = source_components(model, ids)
    batch, heads, positions, width = contributions.shape
    source_mask = torch.zeros((heads, positions), dtype=torch.bool,
                              device=ids.device)
    source_mask.scatter_(1, indices, True)
    def decoded(mask):
        update = attn.out((contributions*mask[None, :, :, None]
                           ).sum(2).reshape(batch, heads*width))
        return F.softmax(model.lm_head(model.final_norm(
            residual+update)).float(), -1)
    native = F.softmax(model(ids)[0][:, -1].float(), -1)
    return dict(full=native, selected=decoded(source_mask),
                removed=decoded(~source_mask))


def kl(p, q):
    return float((p*(p.clamp_min(1e-30).log()-
                     q.clamp_min(1e-30).log())).sum(-1).mean())


def edit_l1(pred, target):
    return float(((pred[1]-pred[0])-(target[1]-target[0])).abs().sum())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--teacher-dir', type=Path, default=Path(
        'language_task_suite/surface_mixture_v2_varied_soft_models'))
    p.add_argument('--run-prefix', type=Path, default=Path(
        'language_task_suite/varied_sparse_sufficiency_broad_'))
    p.add_argument('--arms', default='flat_0,flat_p1,factor_0,factor_p1')
    p.add_argument('--nstreams', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=52102)
    p.add_argument('--max-per-kind', type=int, default=64)
    p.add_argument('--keep', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    meta = json.loads((a.teacher_dir/'results.json').read_text())
    record = next(r for r in meta['records'] if r['depth'] == 2)
    teacher = BilinearGPT(Config(**record['model_config']))
    teacher.load_state_dict(torch.load(a.teacher_dir/'model_depth2.pt',
                                       map_location='cpu', weights_only=True))
    teacher = teacher.to(a.device).eval()
    gen = SurfaceMixtureV2Varied(seed=a.seed, **meta['generator_kwargs'])
    pairs = build_pairs(gen, a.nstreams, a.length)
    fit_keys = set()
    for seed, streams in ((32100, 128), (32101, 512)):
        fit = build_pairs(SurfaceMixtureV2Varied(
            seed=seed, **meta['generator_kwargs']), streams, a.length)
        fit_keys.update((p['kind'], tuple(p['original']),
                         tuple(p['edited'])) for p in fit)
    original_count = len(pairs)
    pairs = [p for p in pairs if (p['kind'], tuple(p['original']),
                                 tuple(p['edited'])) not in fit_keys]
    excluded_count = original_count-len(pairs)
    chosen = defaultdict(list)
    for pair in pairs:
        if len(chosen[pair['kind']]) < a.max_per_kind:
            chosen[pair['kind']].append(pair)
    pairs = [pair for group in chosen.values() for pair in group]
    verb_ids = torch.tensor([gen.token2id[v] for v in VERBS],device=a.device)
    models = {'teacher': teacher}
    for arm_suffix in a.arms.split(','):
        arm = arm_suffix.split('_')[0]
        folder = Path(str(a.run_prefix)+arm_suffix)
        saved_result = json.loads((folder/'results.json').read_text())
        label = next(iter(saved_result['arms']))
        checkpoint = torch.load(folder/f'{label}.pt', map_location='cpu',
                                weights_only=False)
        model = build_students(teacher.cpu(), BilinearGPT, .01)[arm]
        model.load_state_dict(checkpoint['model'])
        models[arm_suffix] = model.to(a.device).eval()
        teacher.to(a.device)
    rows = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        ids = torch.tensor([pair['original'], pair['edited']],device=a.device)
        indices = teacher_indices(teacher, ids, verb_ids, a.keep)
        native = {}
        for label, model in models.items():
            native[label] = branch_probabilities(model, ids, indices)
            # All-source replay was separately verified before fitting.
        reference = native['teacher']['full']
        for label, branches in native.items():
            full = branches['full']
            rows[label][pair['kind']].append(dict(
                native_teacher_kl=kl(reference, full),
                selected_teacher_kl=kl(reference, branches['selected']),
                removed_native_kl=kl(full, branches['removed']),
                native_teacher_edit_l1=edit_l1(full, reference),
                selected_teacher_edit_l1=edit_l1(branches['selected'], reference),
                removed_native_edit_l1=edit_l1(branches['removed'], full)))
    summary = {label: {kind: dict(n=len(items), **{key: float(np.mean(
        [item[key] for item in items])) for key in items[0]})
        for kind, items in kinds.items()} for label, kinds in rows.items()}
    result = dict(config={**vars(a), 'teacher_dir': str(a.teacher_dir),
                          'run_prefix': str(a.run_prefix),
                          'output': str(a.output)},
                  evaluated_counts=dict(Counter(p['kind'] for p in pairs)),
                  fit_overlap_exclusions=excluded_count,
                  metrics=summary,
                  caveat='Teacher top-K gate is selected from both prefixes; '
                         'removed-source KL is against each model native output, '
                         'while selected-source KL is against full teacher. '
                         'Only second-layer final-query source paths are masked.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for label, kinds in summary.items():
        print(label, 'cue selected KL', round(
            kinds['cue']['selected_teacher_kl'], 5),
            'cue removed KL', round(kinds['cue']['removed_native_kl'], 5),
            flush=True)


if __name__ == '__main__':
    main()
