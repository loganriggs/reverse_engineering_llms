"""Iterated attention L1/prune/no-L1 recovery on corpus list+bracket LMs."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_corpus_rule_reference_ov_rank import fidelity
from eval_corpus_rule_reference_transfer import one_kind
from sparsify_original_v3_for_rule_graphs import global_prune, mask_stats, enforce
from train_corpus_delimiter_bilinear import _model
from train_corpus_list_joint_supervised import list_rows


ATTN_PARTS = {'q1', 'k1', 'q2', 'k2', 'v', 'o'}


def attention_parameters(model):
    return {name: param for name, param in model.named_parameters()
            if name.startswith('layers.') and len(name.split('.')) == 4
            and name.split('.')[2] in ATTN_PARTS and name.endswith('.weight')}


def training_data(list_cache, bracket_cache):
    lists = torch.load(list_cache, map_location='cpu', weights_only=True)
    brackets = torch.load(bracket_cache, map_location='cpu', weights_only=True)
    l = list_rows(lists)
    return {'list': {'x': l['x'], 'y': l['y'],
                     'site': l['site'][:, 0]},
            'bracket': {'x': brackets['train']['x'],
                        'y': brackets['train']['y'],
                        'site': brackets['train']['site_position']}}


def sample(data, n, generator, device):
    index = torch.randint(len(data['x']), (n,), generator=generator)
    return {key: value[index].to(device) for key, value in data.items()}


def kl(reference, candidate):
    return F.kl_div(candidate.log_softmax(-1),
                    reference.softmax(-1), reduction='batchmean')


def phase(teacher, student, parameters, masks, data, a, steps,
          generator, l1):
    optimizer = torch.optim.AdamW(parameters.values(), lr=a.lr,
                                  weight_decay=0.)
    losses = []
    for _ in range(steps):
        left = sample(data['list'], a.batch_size//2, generator, a.device)
        right = sample(data['bracket'], a.batch_size//2, generator, a.device)
        x = torch.cat((left['x'], right['x']))
        y = torch.cat((left['y'], right['y']))
        site = torch.cat((left['site'], right['site']))
        ix = torch.arange(len(x), device=a.device)
        with torch.no_grad():
            reference = teacher(x).float()
        output = student(x).float()
        site_loss = kl(reference[ix, site], output[ix, site])
        positions = torch.arange(0, x.shape[1], 32, device=a.device)
        valid = ((y[:, positions] != -100) &
                 (positions[None, :] != site[:, None]))
        background = kl(reference[:, positions][valid],
                        output[:, positions][valid])
        loss = site_loss + a.background_weight*background
        if l1:
            loss = loss + l1*sum(param[ masks[name] ].abs().sum()
                               for name,param in parameters.items())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite sparse-training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name,param in parameters.items():
            param.grad.mul_(masks[name])
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.)
        optimizer.step()
        enforce(parameters, masks)
        losses.append(loss.item())
    return sum(losses[-50:])/min(50, len(losses))


@torch.no_grad()
def evaluate(teacher, student, panels, tokenizer, device):
    result = {}
    for label, panel in panels.items():
        result[label] = {}
        for kind in ('list', 'bracket'):
            rows = [row for row in panel if row['kind'] == kind]
            result[label][kind] = {
                'behavior': one_kind(student, rows, device, 4, tokenizer),
                'fidelity': fidelity(teacher, student, rows, device, 4)}
    return result


def gates(baseline, score):
    result = {}
    for panel in score:
        l0 = baseline[panel]['list']['behavior']
        l = score[panel]['list']['behavior']
        b0 = baseline[panel]['bracket']['behavior']
        b = score[panel]['bracket']['behavior']
        lf, bf = score[panel]['list']['fidelity'], score[panel]['bracket']['fidelity']
        result[panel] = {
            'list_exact_within_5pt': l['paired_top1_fraction'] >=
                                     l0['paired_top1_fraction']-.05,
            'bracket_character_within_5pt':
                b['paired_closer_character_fraction'] >=
                b0['paired_closer_character_fraction']-.05,
            'list_effect_mae_at_most_2': lf['effect_mean_abs_error'] <= 2.,
            'bracket_effect_mae_at_most_2': bf['effect_mean_abs_error'] <= 2.,
            'list_background_kl_at_most_point_1':
                lf['sampled_background_teacher_kl'] <= .1,
            'bracket_background_kl_at_most_point_1':
                bf['sampled_background_teacher_kl'] <= .1,
        }
        result[panel]['all_pass'] = all(result[panel].values())
    return result


def save_checkpoint(path, model, masks, saved, generator):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: saved[key] for key in ('depth', 'd_model', 'heads',
                                           'n_ctx', 'vocab_size', 'norm')}
    payload['state_dict'] = {key: value.detach().cpu()
                             for key,value in model.state_dict().items()}
    payload['masks'] = {key: value.detach().cpu()
                        for key,value in masks.items()}
    payload['sample_rng'] = generator.get_state()
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--test-caches', type=Path, nargs=2, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', default='.4,.6')
    p.add_argument('--steps-per-round', type=int, default=100)
    p.add_argument('--recovery-steps', type=int, default=250)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--l1', type=float, default=1e-5)
    p.add_argument('--background-weight', type=float, default=.25)
    p.add_argument('--sample-seed', type=int, default=84001)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.batch_size % 2:
        raise ValueError('Batch size must be even')
    targets = [float(x) for x in a.targets.split(',')]
    if targets != [.4, .6]:
        raise ValueError('This pilot has predeclared 40%, 60% stages')
    torch.set_num_threads(4)
    data = training_data(a.list_cache, a.bracket_cache)
    panels = {f'panel_{i+1}': torch.load(path, map_location='cpu',
                                        weights_only=True)['records']
              for i,path in enumerate(a.test_caches)}
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    teacher = _model(saved['vocab_size'], saved['n_ctx'], saved['depth'],
                     saved['d_model'], saved['heads'], a.device, saved['norm'])
    teacher.load_state_dict(saved['state_dict'])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    baseline = evaluate(teacher, teacher, panels, tokenizer, a.device)
    report = {'config': {key: [str(v) for v in value] if isinstance(value, list)
                         else str(value) if isinstance(value, Path) else value
                         for key,value in vars(a).items()},
              'baseline': baseline,
              'scope': 'Only 12 q1/k1/q2/k2/v/o attention matrices are '
                       'trainable and pruned; embeddings, MLPs, and '
                       'unembedding are fixed and dense. Train on separate '
                       'list/bracket source windows; fresh ordinary-order '
                       'test panels are never fit. Two arms share sample '
                       'order and prune targets: L1 prephases versus zero-L1 '
                       'prephases. Both get final masked no-L1 recovery. '
                       'Fidelity gates are predeclared 5pt behavior drops, '
                       'effect MAE<=2 logits, sampled background KL<=.1 '
                       'on each panel, not a claim of human legibility.',
              'arms': {}}
    if a.resume and a.output.exists():
        previous = json.loads(a.output.read_text())
        stable = ('checkpoint', 'targets', 'steps_per_round',
                  'recovery_steps', 'batch_size', 'lr', 'l1',
                  'background_weight', 'sample_seed')
        if any(previous['config'][key] != report['config'][key]
               for key in stable):
            raise ValueError('Resume configuration differs')
        report['arms'] = previous['arms']
    for arm, coefficient in (('l1', a.l1), ('no_l1', 0.)):
        student = deepcopy(teacher)
        parameters = attention_parameters(student)
        assert len(parameters) == 12
        for name,param in student.named_parameters():
            param.requires_grad_(name in parameters)
        masks = {name: torch.ones_like(param, dtype=torch.bool)
                 for name,param in parameters.items()}
        generator = torch.Generator(device='cpu').manual_seed(a.sample_seed)
        rows = report['arms'].get(arm, [])
        if rows and rows[-1]['stage'] == 'recovered_no_l1':
            print(arm, 'already complete', flush=True)
            continue
        if rows:
            prior = torch.load(a.output_dir/f"{arm}_{rows[-1]['stage']}.pt",
                               map_location='cpu', weights_only=True)
            student.load_state_dict(prior['state_dict'])
            for name in masks:
                masks[name].copy_(prior['masks'][name].to(a.device))
            generator.set_state(prior['sample_rng'])
            print(arm, 'resuming after', rows[-1]['stage'], flush=True)
        completed_targets = sum(row['stage'].startswith('pruned_')
                                for row in rows)
        for target in targets[completed_targets:]:
            loss = phase(teacher, student, parameters, masks, data, a,
                         a.steps_per_round, generator, coefficient)
            global_prune(parameters, masks, target)
            score = evaluate(teacher, student, panels, tokenizer, a.device)
            stage = f'pruned_{round(100*target)}'
            rows.append({'stage': stage, 'loss_last50': loss,
                         'attention_mask': mask_stats(masks),
                         'scores': score, 'gates': gates(baseline, score)})
            save_checkpoint(a.output_dir/f'{arm}_{stage}.pt',
                            student, masks, saved, generator)
            print(arm, stage, rows[-1]['attention_mask'], flush=True)
            report['arms'][arm] = rows
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, indent=2)+'\n')
        loss = phase(teacher, student, parameters, masks, data, a,
                     a.recovery_steps, generator, 0.)
        score = evaluate(teacher, student, panels, tokenizer, a.device)
        rows.append({'stage': 'recovered_no_l1', 'loss_last50': loss,
                     'attention_mask': mask_stats(masks),
                     'scores': score, 'gates': gates(baseline, score)})
        save_checkpoint(a.output_dir/f'{arm}_recovered_no_l1.pt',
                        student, masks, saved, generator)
        print(arm, 'recovered_no_l1', rows[-1]['attention_mask'], flush=True)
        report['arms'][arm] = rows
        a.output.write_text(json.dumps(report, indent=2)+'\n')
        del student
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
