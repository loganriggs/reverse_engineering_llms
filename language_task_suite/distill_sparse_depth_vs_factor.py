"""Compare sparse projection factors with extra bilinear attention depth.

The 2-layer varied-language teacher is embedded exactly in both 2-layer
factored and 4-layer residual students before sparsification. Only attention
weights are pruned; embedding and unembedding stay dense. This is a pilot for
behavioral fidelity, not a claim that sparse coordinates are interpretable.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from surface_mixture_v2_varied import SurfaceMixtureV2Varied
from surface_mixture_v3 import SurfaceMixtureV3
from train_surface_mixture import evaluate, make_batch, make_eval


class FactoredLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        u, s, vh = torch.linalg.svd(weight.detach().double(), full_matrices=False)
        root = s.sqrt()
        self.first = nn.Linear(weight.shape[1], len(s), bias=False)
        self.second = nn.Linear(len(s), weight.shape[0], bias=False)
        with torch.no_grad():
            self.first.weight.copy_((root[:, None] * vh).to(weight.dtype))
            self.second.weight.copy_((u * root[None, :]).to(weight.dtype))

    @property
    def weight(self):
        return self.second.weight @ self.first.weight

    def forward(self, x):
        return self.second(self.first(x))


def build_students(teacher, BilinearGPT, extra_init_scale):
    cfg = copy.deepcopy(teacher.config)
    factored = copy.deepcopy(teacher)
    for block in factored.blocks:
        for name in ('q1', 'k1', 'q2', 'k2', 'v', 'out'):
            setattr(block.attn, name, FactoredLinear(getattr(block.attn, name).weight))

    deep_cfg = copy.deepcopy(cfg)
    deep_cfg.n_layer = 2 * cfg.n_layer
    deep_cfg.block_has_mlp = (False,) * deep_cfg.n_layer
    torch.manual_seed(1147)
    deep = BilinearGPT(deep_cfg)
    deep.embed.load_state_dict(teacher.embed.state_dict())
    deep.lm_head.load_state_dict(teacher.lm_head.state_dict())
    deep.final_norm.load_state_dict(teacher.final_norm.state_dict())
    for i, block in enumerate(teacher.blocks):
        deep.blocks[2 * i].load_state_dict(block.state_dict())
    # The new blocks have zero output maps by the base model constructor,
    # so their residual updates are exactly identity at initialization.
    perturbed = copy.deepcopy(deep)
    for i in (1, 3):
        with torch.no_grad():
            nn.init.normal_(perturbed.blocks[i].attn.out.weight,
                            std=extra_init_scale / cfg.n_embd**.5)
    split = copy.deepcopy(deep)
    for i, block in enumerate(teacher.blocks):
        for j in (2 * i, 2 * i + 1):
            split.blocks[j].load_state_dict(block.state_dict())
            with torch.no_grad():
                split.blocks[j].attn.out.weight.mul_(.5)
    return {'flat': copy.deepcopy(teacher), 'factor': factored, 'depth_identity': deep,
            'depth_perturbed': perturbed, 'depth_split': split}


def selected(model):
    return {name: p for name, p in model.named_parameters() if '.attn.' in name}


def masks_for(model):
    return {name: torch.ones_like(p) for name, p in selected(model).items()}


def active_count(masks):
    return sum(int(m.sum()) for m in masks.values())


def enforce_masks(model, masks):
    with torch.no_grad():
        for name, p in selected(model).items():
            p.mul_(masks[name])


def prune_to(model, masks, target):
    params = selected(model)
    active = torch.cat([p.detach().abs()[masks[name].bool()]
                        for name, p in params.items()])
    remove = len(active) - target
    if remove <= 0:
        return 0
    # Stable tie-breaking matters because the inserted depth blocks begin with
    # zero outputs. Prune exactly the requested count, not all tied zeros.
    order = torch.argsort(active, stable=True)[:remove]
    flat = torch.zeros(len(active), dtype=torch.bool)
    flat[order] = True
    cursor = 0
    with torch.no_grad():
        for name, p in params.items():
            indices = masks[name].bool().nonzero(as_tuple=True)
            n = len(indices[0])
            discard = tuple(axis[flat[cursor:cursor+n]] for axis in indices)
            masks[name][discard] = 0
            p[discard] = 0
            cursor += n
    assert active_count(masks) == target
    return remove


def prune_to_balanced_matrices(model, masks, target):
    """Keep an equal edge quota in each attention matrix, including new blocks.

    This is a depth-participation control, not a claim that equal allocation is
    an optimal sparse circuit. It prevents global magnitude pruning from
    deleting a small newly initialized output map before it can be trained.
    """
    params = selected(model)
    before = active_count(masks)
    names = list(params)
    if target < len(names):
        raise ValueError('Budget cannot keep one edge per attention matrix')
    quotas = {name: target//len(names)+(i < target % len(names))
              for i, name in enumerate(names)}
    with torch.no_grad():
        for name, p in params.items():
            quota = quotas[name]
            available = masks[name].bool().flatten().nonzero().flatten()
            if quota > len(available):
                raise ValueError(f'Cannot restore pruned entries in {name}')
            if quota == len(available):
                continue
            values = p.detach().flatten()[available].abs()
            keep = available[torch.argsort(values, stable=True)[-quota:]]
            new_mask = torch.zeros_like(p).flatten()
            new_mask[keep] = 1
            masks[name].copy_(new_mask.reshape_as(p))
            p.mul_(masks[name])
    assert active_count(masks) == target
    return before-target


def teacher_kl(student, teacher, x):
    with torch.no_grad():
        target = teacher(x)[0].float()
        log_target = F.log_softmax(target, -1)
        prob_target = log_target.exp()
    logits = student(x)[0].float()
    log_student = F.log_softmax(logits, -1)
    return (prob_target * (log_target - log_student)).sum(-1).mean()


def train_phase(student, teacher, masks, gen, steps, batch, length, lr, l1):
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.)
    params = selected(student)
    student.train()
    for _ in range(steps):
        x, _ = make_batch(gen, batch, length, 'cpu')
        loss = teacher_kl(student, teacher, x)
        if l1:
            loss = loss + l1 * sum((p.abs() * masks[name]).sum()
                                    for name, p in params.items())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite distillation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 10.)
        for name, p in params.items():
            p.grad.mul_(masks[name])
        optimizer.step()
        enforce_masks(student, masks)
    student.eval()


@torch.no_grad()
def panel_scores(student, teacher, panel):
    divergence = []
    for x in panel['x'].split(16):
        divergence.append(teacher_kl(student, teacher, x).item() * len(x))
    result = evaluate(student, panel)
    return dict(teacher_kl=sum(divergence) / len(panel['x']),
                oracle_kl={key: row['kl'] for key, row in result.items()
                           if row['n'] > 0})


def generator_for_teacher(record_file):
    config = record_file['config']
    name = config['generator']
    if name == 'v2_varied':
        return SurfaceMixtureV2Varied, dict(record_file.get('generator_kwargs') or {})
    if name == 'v3':
        kwargs = dict(record_file.get('generator_kwargs') or {})
        if 'object_vocab_size' not in kwargs:
            kwargs['object_vocab_size'] = config.get('object_vocab_size')
        return SurfaceMixtureV3, kwargs
    raise ValueError(f'Unsupported teacher generator: {name}')


def evaluation_panels(generator_class, kwargs, streams, length):
    panels = {'in_distribution': make_eval(
        generator_class(seed=67231, **kwargs), streams, length, 'cpu')}
    if generator_class is SurfaceMixtureV2Varied:
        panels['return_shift'] = make_eval(SurfaceMixtureV2Varied(
            seed=67232, subject_repeat_mass=.1, subject_return_mass=.3,
            **kwargs), streams, length, 'cpu')
        panels['surface_shift'] = make_eval(SurfaceMixtureV2Varied(
            seed=67233, parenthetical_mass=.05, determiner_variation=.9,
            **kwargs), streams, length, 'cpu')
    else:
        size = kwargs.get('object_vocab_size')
        if size is not None:
            shift_kwargs = {**kwargs, 'object_vocab_size': min(32, 2 * size)}
            if shift_kwargs != kwargs:
                panels['object_lexicon_shift'] = make_eval(
                    generator_class(seed=67232, **shift_kwargs),
                    streams, length, 'cpu')
    return panels


def structure_stats(model, masks):
    stats = {}
    for i, block in enumerate(model.blocks):
        names = [name for name in masks if name.startswith(f'blocks.{i}.attn.')]
        active = sum(int(masks[name].sum()) for name in names)
        stats[str(i)] = dict(active_edges=active,
                             output_fro=float(block.attn.out.weight.detach().norm()))
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--teacher-dir', type=Path, default=Path('language_task_suite/surface_mixture_v2_varied_soft_models'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--arms', default='factor,depth_perturbed')
    p.add_argument('--densities', default='.5,.3,.15')
    p.add_argument('--edge-budgets', default='',
                   help='Comma-separated absolute attention-edge budgets; '
                        'overrides --densities for matched architecture comparisons')
    p.add_argument('--steps', type=int, default=80,
                   help='L1 steps before each prune and equal no-L1 steps after')
    p.add_argument('--warmup-steps', type=int, default=120,
                   help='No-L1 distillation before the first prune')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--length', type=int, default=48)
    p.add_argument('--eval-streams', type=int, default=64)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--l1', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--extra-init-scale', type=float, default=.01)
    p.add_argument('--balanced-matrix-prune', action='store_true',
                   help='Reserve an equal edge quota for every attention matrix')
    p.add_argument('--recover-from-dir', type=Path,
                   help='Continue fixed-mask checkpoints with no L1')
    p.add_argument('--recover-budget', type=int, default=29491)
    p.add_argument('--recover-steps', type=int, default=300)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    record_file = json.loads((a.teacher_dir / 'results.json').read_text())
    record = next(r for r in record_file['records'] if r['depth'] == 2)
    teacher = BilinearGPT(Config(**record['model_config']))
    teacher.load_state_dict(torch.load(a.teacher_dir / 'model_depth2.pt',
                                       map_location='cpu', weights_only=True))
    teacher.eval().requires_grad_(False)
    students = build_students(teacher, BilinearGPT, a.extra_init_scale)
    for student in students.values():
        student.requires_grad_(True)
    generator_class, kwargs = generator_for_teacher(record_file)
    panels = evaluation_panels(generator_class, kwargs, a.eval_streams, a.length)
    torch.manual_seed(67234)
    probe = torch.randint(0, teacher.config.vocab_size, (4, 32))
    for arm, student in students.items():
        student.eval()
        with torch.no_grad():
            error = (student(probe)[0] - teacher(probe)[0]).abs().max().item()
        print('initial', arm, 'max logit error', error, flush=True)
        if arm not in ('depth_perturbed', 'depth_split') and error > 2e-4:
            raise AssertionError(f'{arm} initialization does not reproduce teacher')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = {name: panel_scores(teacher, teacher, panel)
                for name, panel in panels.items()}
    config_out = {**vars(a), 'teacher_dir': str(a.teacher_dir),
                  'output_dir': str(a.output_dir),
                  'recover_from_dir': (str(a.recover_from_dir)
                                       if a.recover_from_dir else None)}
    for arm in a.arms.split(','):
        if arm not in students:
            raise ValueError(arm)
        model = students[arm]
        masks = masks_for(model)
        raw = active_count(masks)
        if a.recover_from_dir is not None:
            path = a.recover_from_dir / f'{arm}_{a.recover_budget/raw:.4f}.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'])
            for name, mask in masks.items():
                mask.copy_(saved['masks'][name])
            enforce_masks(model, masks)
            if active_count(masks) != a.recover_budget:
                raise ValueError('Loaded checkpoint has wrong active-edge budget')
            recovery_gen = generator_class(seed=93200+a.seed, **kwargs)
            train_phase(model, teacher, masks, recovery_gen, a.recover_steps,
                        a.batch, a.length, a.lr, 0.)
            row = dict(density=a.recover_budget/raw,
                       active_attention_edges=active_count(masks),
                       total_active_parameters=active_count(masks)+sum(
                           p.numel() for name, p in model.named_parameters()
                           if name not in masks),
                       structure=structure_stats(model, masks),
                       panels={name: panel_scores(model, teacher, panel)
                               for name, panel in panels.items()})
            print('recovered', arm, row['panels'], flush=True)
            torch.save(dict(model=model.state_dict(), masks=masks,
                            config=config_out, density=row['density']),
                       a.output_dir / f'{arm}_{row["density"]:.4f}.pt')
            (a.output_dir / f'{arm}_results.json').write_text(json.dumps(
                dict(config=config_out, teacher_baseline=baseline,
                     rows=[row]), indent=2)+'\n')
            continue
        gen = generator_class(seed=84200+a.seed, **kwargs)
        rows = []
        if a.warmup_steps and arm in ('depth_split', 'depth_perturbed'):
            warmup_gen = generator_class(seed=83200+a.seed, **kwargs)
            train_phase(model, teacher, masks, warmup_gen, a.warmup_steps, a.batch,
                        a.length, a.lr, 0.)
        targets = ([int(s) for s in a.edge_budgets.split(',')]
                   if a.edge_budgets else
                   [round(float(s) * raw) for s in a.densities.split(',')])
        if any(target <= 0 or target > raw for target in targets):
            raise ValueError('Each edge budget must be positive and at most raw edges')
        targets = [target for target in targets if target < raw]
        if targets != sorted(set(targets), reverse=True):
            raise ValueError('Edge budgets must be strictly descending')
        for target in [raw] + targets:
            density = target / raw
            if target != raw:
                train_phase(model, teacher, masks, gen, a.steps, a.batch,
                            a.length, a.lr, a.l1)
                if a.balanced_matrix_prune:
                    prune_to_balanced_matrices(model, masks, target)
                else:
                    prune_to(model, masks, target)
                train_phase(model, teacher, masks, gen, a.steps, a.batch,
                            a.length, a.lr, 0.)
            row = dict(density=density, active_attention_edges=active_count(masks),
                       total_active_parameters=active_count(masks)+sum(
                           p.numel() for name, p in model.named_parameters()
                           if name not in masks),
                       structure=structure_stats(model, masks),
                       panels={name: panel_scores(model, teacher, panel)
                               for name, panel in panels.items()})
            rows.append(row)
            print(arm, density, row['panels'], flush=True)
            torch.save(dict(model=model.state_dict(), masks=masks,
                            config=config_out, density=density),
                       a.output_dir / f'{arm}_{density:.4f}.pt')
            (a.output_dir / f'{arm}_results.json').write_text(json.dumps(
                dict(config=config_out, teacher_baseline=baseline,
                     rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
