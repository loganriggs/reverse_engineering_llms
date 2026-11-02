"""Data-free sparse factor fit for the varied-language teacher's attention maps.

Fit each original projection W by two sparse factors W ~= second @ first.
Global magnitude pruning sets a shared physical edge budget; normalized
matrix-reconstruction loss and gauge balancing then optimize nonzero values.
This is a weight-only baseline, not a reproduction of published SWD/DSF.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from distill_sparse_depth_vs_factor import (active_count, build_students,
                                            enforce_masks, evaluation_panels,
                                            generator_for_teacher, masks_for,
                                            panel_scores, prune_to)


PROJECTIONS = ('q1', 'k1', 'q2', 'k2', 'v', 'out')


def projection_pairs(student, teacher):
    pairs = []
    for student_block, teacher_block in zip(student.blocks, teacher.blocks):
        for name in PROJECTIONS:
            factor = getattr(student_block.attn, name)
            target = getattr(teacher_block.attn, name).weight.detach()
            pairs.append((factor, target))
    return pairs


def relative_matrix_loss(pairs):
    return sum(((factor.weight - target).square().sum() /
                target.square().sum().clamp_min(1e-12))
               for factor, target in pairs) / len(pairs)


@torch.no_grad()
def balance_factors(pairs):
    for factor, _ in pairs:
        first, second = factor.first.weight, factor.second.weight
        rows = first.norm(dim=1)
        cols = second.norm(dim=0)
        live = (rows > 1e-10) & (cols > 1e-10)
        scale = torch.ones_like(rows)
        scale[live] = (cols[live] / rows[live]).sqrt()
        first.mul_(scale[:, None])
        second.div_(scale[None, :])


def fit_stage(student, pairs, masks, steps, lr):
    params = {name: p for name, p in student.named_parameters()
              if name in masks}
    opt = torch.optim.Adam(params.values(), lr=lr)
    for step in range(steps):
        loss = relative_matrix_loss(pairs)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite sparse matrix reconstruction loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in params.items():
            param.grad.mul_(masks[name])
        opt.step()
        enforce_masks(student, masks)
        if (step + 1) % 20 == 0:
            balance_factors(pairs)
            enforce_masks(student, masks)
    return float(relative_matrix_loss(pairs).detach())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--teacher-dir', type=Path,
                   default=Path('language_task_suite/surface_mixture_v2_varied_soft_models'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--edge-budgets', default='68813,49152,39322,29491')
    p.add_argument('--steps-per-budget', type=int, default=250)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--eval-streams', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    data = json.loads((a.teacher_dir / 'results.json').read_text())
    record = next(r for r in data['records'] if r['depth'] == 2)
    teacher = BilinearGPT(Config(**record['model_config']))
    teacher.load_state_dict(torch.load(a.teacher_dir / 'model_depth2.pt',
                                       map_location='cpu', weights_only=True))
    teacher.eval().requires_grad_(False)
    student = build_students(teacher, BilinearGPT, .01)['factor']
    student.requires_grad_(True).eval()
    pairs = projection_pairs(student, teacher)
    masks = masks_for(student)
    raw = active_count(masks)
    targets = [int(v) for v in a.edge_budgets.split(',')]
    if targets != sorted(set(targets), reverse=True) or any(
            target <= 0 or target >= raw for target in targets):
        raise ValueError('Budgets must be strictly descending below raw count')
    generator_class, kwargs = generator_for_teacher(data)
    panels = evaluation_panels(generator_class, kwargs,
                               a.eval_streams, a.length)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in targets:
        removed = prune_to(student, masks, target)
        fit_loss = fit_stage(student, pairs, masks, a.steps_per_budget, a.lr)
        row = dict(active_attention_edges=active_count(masks),
                   density=target/raw, removed=removed,
                   mean_relative_matrix_mse=fit_loss,
                   mean_relative_matrix_error=fit_loss**.5,
                   panels={name: panel_scores(student, teacher, panel)
                           for name, panel in panels.items()})
        rows.append(row)
        print(row, flush=True)
        torch.save(dict(model=student.state_dict(), masks=masks,
                        density=target/raw),
                   a.output_dir / f'factor_{target/raw:.4f}.pt')
        (a.output_dir / 'factor_results.json').write_text(json.dumps(
            dict(config={**vars(a), 'teacher_dir': str(a.teacher_dir),
                         'output_dir': str(a.output_dir)},
                 rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
