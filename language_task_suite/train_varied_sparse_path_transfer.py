"""Pilot: does teacher source-path supervision help fixed-mask sparse students?

Only second-block paths and verb-output contrasts are supervised. One entire
edit family is withheld from this paired objective, not from language KL or
the teacher's original training. This is a gauge-anchored teacher-student test.
"""

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from distill_sparse_depth_vs_factor import (build_students, enforce_masks,
                                            selected, teacher_kl)
from eval_surface_mixture_edits import predict, score
from eval_surface_mixture_v2_edits import build_pairs
from surface_mixture_v2 import VERBS
from surface_mixture_v2_varied import SurfaceMixtureV2Varied
from train_surface_mixture import make_batch


def path_signatures(model, ids, verb_ids):
    """Differentiable, exact second-layer source contributions in logit space.

    The output is a centered 12-verb linear contrast before final RMSNorm;
    this deliberately does not claim to reproduce probability effects.
    """
    from model import apply_rotary_emb

    residual = model.blocks[0](model.embed(ids))
    attn = model.blocks[1].attn
    x = model.blocks[1].norm_attn(residual)
    b, t, _ = x.shape
    h, d = attn.n_head, attn.head_dim
    projected = {name: getattr(attn, name)(x).reshape(b, t, h, d)
                 for name in ('q1', 'k1', 'q2', 'k2', 'v')}
    cos, sin = attn.rotary(projected['q1'])
    scores = []
    for branch in (1, 2):
        q = apply_rotary_emb(projected[f'q{branch}'], cos, sin)
        k = apply_rotary_emb(projected[f'k{branch}'], cos, sin)
        scores.append(torch.einsum('bihd,bjhd->bhij', q, k)[:, :, -1]/d)
    contributions = (scores[0]*scores[1]).unsqueeze(-1)*(
        projected['v'].permute(0, 2, 1, 3))
    out = attn.out.weight.reshape(model.config.n_embd, h, d)
    rows = model.lm_head.weight[verb_ids]
    contrast = rows-rows.mean(0, keepdim=True)
    path = torch.einsum('bhtd,chd,vc->bhtv', contributions, out, contrast)
    return path


def paired_path_loss(student, teacher, pair, verb_ids, keep):
    ids = torch.tensor([pair['original'], pair['edited']],
                       device=verb_ids.device)
    with torch.no_grad():
        target = path_signatures(teacher, ids, verb_ids)
        importance = target.square().sum(-1).amax(0)
        indices = importance.topk(min(keep, ids.shape[1]), dim=-1).indices
        expanded = indices.unsqueeze(0).unsqueeze(-1).expand(
            2, -1, -1, target.shape[-1])
        target_edit = (target[1]-target[0]).gather(1, expanded[0])
    student_path = path_signatures(student, ids, verb_ids)
    student_edit = (student_path[1]-student_path[0]).gather(1, expanded[0])
    relative = (student_edit-target_edit).square().sum()/(
        target_edit.square().sum().clamp_min(1e-4))
    return relative


@torch.no_grad()
def panel_kl(model, teacher, gen, nstreams, length):
    weighted = 0.
    count = 0
    for _ in range((nstreams+15)//16):
        x, _ = make_batch(gen, min(16, nstreams-count), length, 'cpu')
        x = x.to(next(model.parameters()).device)
        weighted += float(teacher_kl(model, teacher, x))*len(x)
        count += len(x)
    return weighted/count


@torch.no_grad()
def pair_scores(model, teacher, pairs, gen):
    model.eval().cpu()
    before = predict(model, [p['original'] for p in pairs], len(gen.vocab))
    after = predict(model, [p['edited'] for p in pairs], len(gen.vocab))
    teacher_before = predict(teacher.cpu(),
                             [p['original'] for p in pairs], len(gen.vocab))
    teacher_after = predict(teacher,
                            [p['edited'] for p in pairs], len(gen.vocab))
    result = score(pairs, before, after, gen)
    delta = (after-before)-(teacher_after-teacher_before)
    for kind in result:
        ix = [i for i, p in enumerate(pairs) if p['kind'] == kind]
        result[kind]['teacher_effect_l1'] = float(np.abs(delta[ix]).sum(-1).mean())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--teacher-dir', type=Path, default=Path(
        'language_task_suite/surface_mixture_v2_varied_soft_models'))
    p.add_argument('--base-dir', type=Path, default=Path(
        'language_task_suite/sparse_architecture_recovery_pilot'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--arms', default='flat,factor')
    p.add_argument('--path-weights', default='0,.001,.01')
    p.add_argument('--holdout-kind', default='cue')
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--fit-streams', type=int, default=128)
    p.add_argument('--test-streams', type=int, default=128)
    p.add_argument('--keep', type=int, default=4)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    meta = json.loads((a.teacher_dir/'results.json').read_text())
    assert meta['config']['generator'] == 'v2_varied'
    record = next(r for r in meta['records'] if r['depth'] == 2)
    teacher = BilinearGPT(Config(**record['model_config']))
    teacher.load_state_dict(torch.load(a.teacher_dir/'model_depth2.pt',
                                       map_location='cpu', weights_only=True))
    teacher.eval().requires_grad_(False)
    kwargs = meta['generator_kwargs']
    fit_gen = SurfaceMixtureV2Varied(seed=32100+a.seed, **kwargs)
    fit = build_pairs(fit_gen, a.fit_streams, a.length)
    train_pairs = defaultdict(list)
    for pair in fit:
        if pair['kind'] not in (a.holdout_kind, 'nuisance_adjective',
                                'nuisance_determiner'):
            train_pairs[pair['kind']].append(pair)
    assert len(train_pairs) >= 2 and all(train_pairs.values())
    test_gen = SurfaceMixtureV2Varied(seed=42100+a.seed, **kwargs)
    test = build_pairs(test_gen, a.test_streams, a.length)
    fit_keys = {(p['kind'], tuple(p['original']), tuple(p['edited'])) for p in fit}
    test = [p for p in test if (p['kind'], tuple(p['original']),
                               tuple(p['edited'])) not in fit_keys]
    verb_ids = torch.tensor([fit_gen.token2id[v] for v in VERBS], device=a.device)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    results = dict(config={**vars(a), 'teacher_dir': str(a.teacher_dir),
                           'base_dir': str(a.base_dir),
                           'output_dir': str(a.output_dir)},
                   fit_counts=dict(Counter(p['kind'] for p in fit)),
                   test_counts=dict(Counter(p['kind'] for p in test)),
                   path_fit_kinds=sorted(train_pairs),
                   holdout_scope='paired path loss only; ordinary language KL '
                                 'and source teacher include all rule families',
                   arms={})
    rng = np.random.default_rng(53100+a.seed)
    schedule = [[train_pairs[k][int(rng.integers(len(train_pairs[k])))]
                 for k in sorted(train_pairs)] for _ in range(a.steps)]
    for arm in a.arms.split(','):
        if arm not in ('flat', 'factor'):
            raise ValueError(arm)
        raw = 49152 if arm == 'flat' else 98304
        base = torch.load(a.base_dir/f'{arm}_{29491/raw:.4f}.pt',
                          map_location='cpu', weights_only=False)
        for weight in [float(s) for s in a.path_weights.split(',')]:
            model = build_students(teacher, BilinearGPT, .01)[arm]
            model.load_state_dict(copy.deepcopy(base['model']))
            masks = {name: mask.to(a.device).clone()
                     for name, mask in base['masks'].items()}
            model = model.to(a.device).train().requires_grad_(True)
            teacher = teacher.to(a.device).eval()
            optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                          weight_decay=0.)
            language_gen = SurfaceMixtureV2Varied(seed=62100+a.seed, **kwargs)
            history = []
            for step, picked in enumerate(schedule, 1):
                x, _ = make_batch(language_gen, a.batch, a.length, 'cpu')
                kl = teacher_kl(model, teacher, x.to(a.device))
                path = torch.zeros((), device=a.device)
                if weight:
                    path = torch.stack([paired_path_loss(
                        model, teacher, pair, verb_ids, a.keep)
                        for pair in picked]).mean()
                loss = kl+weight*path
                if not torch.isfinite(loss):
                    raise RuntimeError((arm, weight, step, float(loss)))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                for name, param in selected(model).items():
                    param.grad.mul_(masks[name])
                optimizer.step()
                enforce_masks(model, masks)
                if step in (1, a.steps) or step % 50 == 0:
                    history.append(dict(step=step, kl=float(kl.detach()),
                                        path_relative=float(path.detach())))
            model.eval()
            eval_gen = SurfaceMixtureV2Varied(seed=72100+a.seed, **kwargs)
            in_distribution_kl = panel_kl(model, teacher, eval_gen,
                                          a.test_streams, a.length)
            path_by_kind = defaultdict(list)
            for pair in test:
                if len(path_by_kind[pair['kind']]) < 24:
                    with torch.no_grad():
                        value = paired_path_loss(model, teacher, pair,
                                                 verb_ids, a.keep)
                    path_by_kind[pair['kind']].append(float(value))
            path_test = {kind: dict(n=len(values), mean_relative_error=float(
                np.mean(values))) for kind, values in path_by_kind.items()}
            model.cpu()
            teacher.cpu()
            edits = pair_scores(model, teacher, test, test_gen)
            label=f'{arm}_path{weight:g}'
            results['arms'][label] = dict(history=history,
                teacher_kl=in_distribution_kl, edits=edits,
                path_test=path_test,
                active_attention_edges=sum(int(mask.sum()) for mask in masks.values()))
            torch.save(dict(model=model.state_dict(),
                            masks={k:v.cpu() for k,v in masks.items()}),
                       a.output_dir/f'{label}.pt')
            (a.output_dir/'results.json').write_text(json.dumps(results, indent=2)+'\n')
            print(label, 'KL', round(in_distribution_kl, 5),
                  'holdout effect', round(edits[a.holdout_kind]['teacher_effect_l1'], 4),
                  flush=True)


if __name__ == '__main__':
    main()
