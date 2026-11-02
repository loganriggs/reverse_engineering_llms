"""Test task-balanced causal-effect supervision on fixed-mask sparse students.

The omitted edit family is held out only from paired-effect supervision. The
teacher and initial sparse students were trained on the full language mixture.
"""

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from distill_sparse_depth_vs_factor import (build_students, enforce_masks,
                                            generator_for_teacher, selected,
                                            teacher_kl)
from eval_sparse_architecture_causal_edits import build_object_pairs, build_verb_pairs
from eval_surface_mixture_edits import predict, score
from surface_mixture_v3 import SurfaceMixtureV3
from train_surface_mixture import make_batch


def make_pairs(generator_class, kwargs, seed, nstreams, length):
    gen = generator_class(seed=seed, **kwargs)
    pairs = build_verb_pairs(gen, nstreams, length)
    if generator_class is SurfaceMixtureV3:
        pairs += build_object_pairs(generator_class(seed=seed, **kwargs),
                                    nstreams, length)
    return gen, pairs


def unique_pairs(rows, excluded=()):
    seen = set(excluded)
    unique = []
    for row in rows:
        key = (row['kind'], tuple(row['original']), tuple(row['edited']))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique, seen


def pad_batch(rows):
    lengths = torch.tensor([len(row) for row in rows])
    x = torch.zeros(len(rows), int(lengths.max()), dtype=torch.long)
    for i, row in enumerate(rows):
        x[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    return x, lengths-1


def last_probabilities(model, rows):
    x, at = pad_batch(rows)
    logits = model(x)[0]
    return F.softmax(logits[torch.arange(len(rows)), at].float(), -1)


def target_effects(teacher, pairs, vocab_size):
    before = torch.from_numpy(predict(
        teacher, [pair['original'] for pair in pairs], vocab_size))
    after = torch.from_numpy(predict(
        teacher, [pair['edited'] for pair in pairs], vocab_size))
    return after-before


@torch.no_grad()
def effect_scores(model, teacher, pairs, gen, vocab_size):
    original = [pair['original'] for pair in pairs]
    edited = [pair['edited'] for pair in pairs]
    student_before = predict(model, original, vocab_size)
    student_after = predict(model, edited, vocab_size)
    teacher_before = predict(teacher, original, vocab_size)
    teacher_after = predict(teacher, edited, vocab_size)
    oracle = score(pairs, student_before, student_after, gen)
    teacher_effect = teacher_after-teacher_before
    student_effect = student_after-student_before
    for kind in sorted({p['kind'] for p in pairs}):
        ix = np.array([i for i, p in enumerate(pairs) if p['kind'] == kind])
        oracle[kind]['teacher_effect_l1_error'] = float(np.abs(
            teacher_effect[ix]-student_effect[ix]).sum(-1).mean())
    return oracle


def train(model, teacher, masks, generator_class, kwargs, pairs, targets,
          steps, batch, length, seed, lr, effect_weight):
    by_kind = {kind: [i for i, row in enumerate(pairs) if row['kind'] == kind]
               for kind in sorted({row['kind'] for row in pairs})}
    assert all(by_kind.values())
    rng = torch.Generator().manual_seed(10000+seed)
    gen = generator_class(seed=11000+seed, **kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    parameters = selected(model)
    history = []
    model.train()
    for step in range(1, steps+1):
        x, _ = make_batch(gen, batch, length, 'cpu')
        kl = teacher_kl(model, teacher, x)
        effect_loss = torch.zeros((), dtype=kl.dtype)
        if effect_weight:
            ix = [ids[int(torch.randint(len(ids), (1,), generator=rng))]
                  for ids in by_kind.values()]
            picked = [pairs[i] for i in ix]
            before = last_probabilities(model,
                                        [row['original'] for row in picked])
            after = last_probabilities(model,
                                       [row['edited'] for row in picked])
            effect_loss = ((after-before)-targets[ix]).abs().sum(-1).mean()
        loss = kl+effect_weight*effect_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        for name, parameter in parameters.items():
            parameter.grad.mul_(masks[name])
        optimizer.step()
        enforce_masks(model, masks)
        if step == 1 or step % 100 == 0 or step == steps:
            history.append(dict(step=step, ordinary_kl=float(kl.detach()),
                                paired_effect_l1=float(effect_loss.detach()),
                                objective=float(loss.detach())))
    model.eval()
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--teacher-dir', type=Path,
                   default=Path('language_task_suite/surface_mixture_v3_object8'))
    p.add_argument('--flat-dir', type=Path,
                   default=Path('language_task_suite/v3_object8_sparse_recovery'))
    p.add_argument('--factor-dir', type=Path,
                   default=Path('language_task_suite/v3_object8_sparse_recovery'))
    p.add_argument('--holdout-kind', default='object_event_skip')
    p.add_argument('--effect-weight', type=float, default=.1)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--fit-streams', type=int, default=128)
    p.add_argument('--val-streams', type=int, default=64)
    p.add_argument('--test-streams', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--test-seed', type=int, default=None,
                   help='Independent generator seed for the edit test panel')
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    saved_teacher = json.loads((a.teacher_dir/'results.json').read_text())
    record = next(r for r in saved_teacher['records'] if r['depth'] == 2)
    teacher = BilinearGPT(Config(**record['model_config']))
    teacher.load_state_dict(torch.load(a.teacher_dir/'model_depth2.pt',
                                       map_location='cpu', weights_only=True))
    teacher.eval().requires_grad_(False)
    generator_class, kwargs = generator_for_teacher(saved_teacher)
    fit_gen, sampled_fit = make_pairs(generator_class, kwargs, 12000+a.seed,
                                  a.fit_streams, a.length)
    val_gen, sampled_val = make_pairs(generator_class, kwargs, 13000+a.seed,
                                  a.val_streams, a.length)
    test_seed = 14000+a.seed if a.test_seed is None else a.test_seed
    test_gen, sampled_test = make_pairs(generator_class, kwargs, test_seed,
                                a.test_streams, a.length)
    all_fit, fit_keys = unique_pairs(sampled_fit)
    all_val, val_keys = unique_pairs(sampled_val, fit_keys)
    test, _ = unique_pairs(sampled_test, val_keys)
    assert any(p['kind'] == a.holdout_kind for p in all_fit)
    fit = [p for p in all_fit if p['kind'] != a.holdout_kind]
    val = [p for p in all_val if p['kind'] != a.holdout_kind]
    assert {p['kind'] for p in fit} == {p['kind'] for p in val}
    vocab_size = teacher.config.vocab_size
    target = target_effects(teacher, fit, vocab_size)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    record_out = dict(config={**vars(a),
                              'teacher_dir': str(a.teacher_dir),
                              'flat_dir': str(a.flat_dir),
                              'factor_dir': str(a.factor_dir),
                              'output_dir': str(a.output_dir)},
                      fit_pair_counts=dict(Counter(p['kind'] for p in fit)),
                      val_pair_counts=dict(Counter(p['kind'] for p in val)),
                      test_pair_counts=dict(Counter(p['kind'] for p in test)),
                      duplicate_pair_exclusions=dict(
                          fit=len(sampled_fit)-len(all_fit),
                          val=len(sampled_val)-len(all_val),
                          test=len(sampled_test)-len(test)),
                      holdout_scope='paired-effect supervision only; base language '
                                    'training and teacher already include the family',
                      arms={})
    for arm, folder, raw in [('flat', a.flat_dir, 49152),
                              ('factor', a.factor_dir, 98304)]:
        base = torch.load(folder/f'{arm}_{29491/raw:.4f}.pt',
                          map_location='cpu', weights_only=False)
        for objective, weight in [('kl', 0.), ('causal', a.effect_weight)]:
            model = build_students(teacher, BilinearGPT, .01)[arm]
            model.load_state_dict(copy.deepcopy(base['model']))
            masks = {name: mask.clone() for name, mask in base['masks'].items()}
            model.requires_grad_(True)
            history = train(model, teacher, masks, generator_class, kwargs,
                            fit, target, a.steps, a.batch, a.length,
                            a.seed, a.lr, weight)
            model.eval()
            with torch.no_grad():
                validation = effect_scores(model, teacher, val, val_gen,
                                           vocab_size)
                testing = effect_scores(model, teacher, test, test_gen,
                                        vocab_size)
            label = f'{arm}_{objective}'
            record_out['arms'][label] = dict(history=history,
                                            validation=validation,
                                            test=testing,
                                            active_attention_edges=sum(
                                                int(mask.sum()) for mask in masks.values()))
            torch.save(dict(model=model.state_dict(), masks=masks,
                            config=record_out['config']),
                       a.output_dir/f'{label}.pt')
            (a.output_dir/'results.json').write_text(json.dumps(
                record_out, indent=2)+'\n')
            print(label, 'holdout teacher-effect L1',
                  testing[a.holdout_kind]['teacher_effect_l1_error'],
                  'holdout oracle-effect L1',
                  testing[a.holdout_kind]['effect_l1_error'], flush=True)


if __name__ == '__main__':
    main()
