"""Iterated L1 -> magnitude prune -> revert/finetune on a trained bilinear LM.

This is a functional sparsity baseline. It does not by itself identify QK/OV paths.
Selection uses one frozen exact-oracle panel; fresh panels are reported only at end.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from surface_mixture_v2_varied import SurfaceMixtureV2Varied
from train_surface_mixture import evaluate, make_eval


def selected_parameters(model, arm):
    return {name: p for name, p in model.named_parameters()
            if ('.attn.' in name or
                (arm == 'all_linear' and name in ('embed.weight',
                                                   'lm_head.weight')))}


def snapshot(model, masks):
    return (copy.deepcopy(model.state_dict()),
            {name: mask.clone() for name, mask in masks.items()})


def restore(model, masks, saved):
    model.load_state_dict(saved[0])
    for name, mask in masks.items():
        mask.copy_(saved[1][name])


def density(masks):
    return sum(int(mask.sum()) for mask in masks.values()) / sum(
        mask.numel() for mask in masks.values())


def prune(parameters, masks, fraction):
    remaining = torch.cat([param.detach().abs()[masks[name].bool()]
                           for name, param in parameters.items()])
    cut = int(fraction*len(remaining))
    if cut == 0:
        return 0
    threshold = torch.kthvalue(remaining, cut).values
    removed = 0
    with torch.no_grad():
        for name, param in parameters.items():
            discard = masks[name].bool() & (param.abs() <= threshold)
            removed += int(discard.sum())
            masks[name][discard] = 0
            param[discard] = 0
    return removed


def train_steps(model, parameters, masks, generator, steps, batch, length,
                lr, l1, verb_soft_weight):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        panel = make_eval(generator, batch, length, 'cpu')
        logits = model(panel['x'])[0].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               panel['y'].reshape(-1))
        if verb_soft_weight:
            logp = F.log_softmax(logits, -1)
            expected_ce = -(panel['q']*logp).sum(-1)
            loss += verb_soft_weight*expected_ce[panel['verb']].mean()
        if l1:
            loss += l1*sum((param.abs()*masks[name]).sum()
                           for name, param in parameters.items())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite sparsification loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        for name, param in parameters.items():
            param.grad.mul_(masks[name])
        optimizer.step()
        with torch.no_grad():
            for name, param in parameters.items():
                param.mul_(masks[name])
    model.eval()


def score(model, panel):
    scores = evaluate(model, panel)
    return {key: scores[key]['kl'] for key in ('all', 'verb',
                                               'verb_with_induction')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path,
                   default=Path('language_task_suite/surface_mixture_v2_varied_soft_models'))
    p.add_argument('--initial-state-dir', type=Path,
                   help='Continue from a pruned model and its hard masks')
    p.add_argument('--arm', choices=('attention_only', 'all_linear'), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=12)
    p.add_argument('--steps-per-iteration', type=int, default=200)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--prune-fraction', type=float, default=.15)
    p.add_argument('--l1', type=float, default=3e-5)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--verb-soft-weight', type=float, default=2.)
    p.add_argument('--max-all-kl-rise', type=float, default=.02)
    p.add_argument('--max-verb-kl-rise', type=float, default=.025)
    p.add_argument('--max-induction-kl-rise', type=float, default=.035)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', action='store_true',
                   help='Resume the last completed prune round in output-dir')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    saved = json.loads((a.model_dir/'results.json').read_text())
    assert saved['config']['generator'] == 'v2_varied'
    record = next(row for row in saved['records'] if row['depth'] == 2)
    kwargs = dict(saved['generator_kwargs'])
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     map_location='cpu', weights_only=True))
    parameters = selected_parameters(model, a.arm)
    masks = {name: torch.ones_like(param) for name, param in parameters.items()}
    selection = make_eval(SurfaceMixtureV2Varied(seed=20261125, **kwargs),
                          512, a.length, 'cpu')
    baseline = score(model, selection)
    limits = dict(all=baseline['all']+a.max_all_kl_rise,
                  verb=baseline['verb']+a.max_verb_kl_rise,
                  verb_with_induction=(baseline['verb_with_induction']+
                                       a.max_induction_kl_rise))
    if a.initial_state_dir is not None:
        model.load_state_dict(torch.load(
            a.initial_state_dir/'model_depth2.pt',
            map_location='cpu', weights_only=True))
        initial_masks = torch.load(a.initial_state_dir/'masks.pt',
                                   map_location='cpu', weights_only=True)
        if set(initial_masks) != set(masks):
            raise ValueError('Initial mask keys do not match the selected arm')
        for name, mask in masks.items():
            mask.copy_(initial_masks[name])
            if torch.any(parameters[name].detach()[mask == 0] != 0):
                raise ValueError(f'Masked parameter is nonzero: {name}')
    generator = SurfaceMixtureV2Varied(seed=8500+a.seed, **kwargs)
    good = snapshot(model, masks)
    history = [dict(iteration=-1, phase='initial', density=density(masks),
                    scores=score(model, selection))]
    first_iteration = 0
    checkpoint_path = a.output_dir/'round_checkpoint.pt'
    protocol = dict(arm=a.arm, iterations=a.iterations,
                    steps_per_iteration=a.steps_per_iteration,
                    batch=a.batch, length=a.length,
                    prune_fraction=a.prune_fraction, l1=a.l1, lr=a.lr,
                    verb_soft_weight=a.verb_soft_weight, seed=a.seed,
                    max_all_kl_rise=a.max_all_kl_rise,
                    max_verb_kl_rise=a.max_verb_kl_rise,
                    max_induction_kl_rise=a.max_induction_kl_rise,
                    model_dir=str(a.model_dir),
                    initial_state_dir=str(a.initial_state_dir))
    if a.resume:
        checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                weights_only=False)
        if checkpoint['protocol'] != protocol:
            raise ValueError('Resume protocol differs from saved checkpoint')
        model.load_state_dict(checkpoint['model_state'])
        for name, mask in masks.items():
            mask.copy_(checkpoint['masks'][name])
        good = checkpoint['good']
        history = checkpoint['history']
        generator.rng.setstate(checkpoint['generator_rng_state'])
        first_iteration = checkpoint['next_iteration']
        print(a.arm, 'resumed at iteration', first_iteration, flush=True)
    for iteration in range(first_iteration, a.iterations):
        train_steps(model, parameters, masks, generator, a.steps_per_iteration,
                    a.batch, a.length, a.lr, a.l1, a.verb_soft_weight)
        current = score(model, selection)
        row = dict(iteration=iteration, phase='l1', density=density(masks),
                   scores=current)
        history.append(row)
        print(a.arm, row, flush=True)
        if any(current[key] > limit for key, limit in limits.items()):
            restore(model, masks, good)
            train_steps(model, parameters, masks, generator,
                        a.steps_per_iteration, a.batch, a.length, a.lr,
                        0., a.verb_soft_weight)
            current = score(model, selection)
            history.append(dict(iteration=iteration, phase='revert_finetune',
                                density=density(masks), scores=current))
            print(a.arm, 'reverted', history[-1], flush=True)
            break
        good = snapshot(model, masks)
        removed = prune(parameters, masks, a.prune_fraction)
        print(a.arm, 'pruned', removed, 'density', density(masks), flush=True)
        state = dict(protocol=protocol, model_state=copy.deepcopy(model.state_dict()),
                     masks={name: mask.clone() for name, mask in masks.items()},
                     good=good, history=history,
                     generator_rng_state=generator.rng.getstate(),
                     next_iteration=iteration+1)
        temporary = checkpoint_path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)
        if not removed:
            break
    else:
        restore(model, masks, good)
        train_steps(model, parameters, masks, generator,
                    a.steps_per_iteration, a.batch, a.length, a.lr,
                    0., a.verb_soft_weight)
        history.append(dict(iteration=a.iterations, phase='final_finetune',
                            density=density(masks), scores=score(model, selection)))

    fresh = make_eval(SurfaceMixtureV2Varied(seed=20261126, **kwargs),
                      512, a.length, 'cpu')
    low_kwargs = {**kwargs, 'parenthetical_mass': .1}
    low = make_eval(SurfaceMixtureV2Varied(seed=20261127, **low_kwargs),
                    512, a.length, 'cpu')
    output = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'initial_state_dir': str(a.initial_state_dir),
                          'output_dir': str(a.output_dir)},
                  generator_kwargs=kwargs, baseline=baseline, limits=limits,
                  history=history, final_density=density(masks),
                  selected_parameter_counts={name: int(mask.numel())
                                             for name, mask in masks.items()},
                  final_per_matrix_density={name: float(mask.mean())
                                            for name, mask in masks.items()},
                  selection=score(model, selection), fresh=score(model, fresh),
                  low_parenthetical=score(model, low),
                  fresh_baseline=None,
                  caveat='Selection is based on a frozen oracle-KL panel. Fresh '
                         'panels are evaluated after selection. This establishes '
                         'behavioral sparsity, not path interpretation.')
    original = BilinearGPT(Config(**record['model_config']))
    original.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                        map_location='cpu', weights_only=True))
    output['fresh_baseline'] = score(original, fresh)
    output['low_parenthetical_baseline'] = score(original, low)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.output_dir/'model_depth2.pt')
    torch.save(masks, a.output_dir/'masks.pt')
    (a.output_dir/'pruning_results.json').write_text(json.dumps(output, indent=2)+'\n')
    # Preserve the ordinary evaluator's checkpoint format for causal-edit audits.
    (a.output_dir/'results.json').write_text(json.dumps(dict(
        config={'generator': 'v2_varied'}, generator_kwargs=kwargs,
        records=[{'depth': 2, 'model_config': record['model_config']}]),
        indent=2)+'\n')
    checkpoint_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
