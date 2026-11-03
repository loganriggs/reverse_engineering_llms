"""Data-free and score-only-data OV rank reductions with held-out rule transfer.

The object is the exact linear value-output map O_h V_h of one attention
head. QK routing and the rest of the model remain native. Activation fit
uses score-only train-group documents with no inserted player-copy task.
The strict fit arm also removes empirical carrier prose that may mention stats.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from sports_report_carrier_dgp import sample_plan as score_only_plan, realize as score_only_realize
from train_sports_report_multirule import DeepModel, SPECS, draw, stack


KINDS = ('winning_score', 'player_points')
PAD = 50256


def load_model(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    c = saved['config']
    model = DeepModel(50257, c['d_model'], c['heads'],
                      SPECS[f"block{c['depth']}"], c['n_ctx'],
                      norm='rms', residual='lerp',
                      attention='bilinear').to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, c


@torch.no_grad()
def score_only_covariance(model, tokenizer, n_docs, seed, n_ctx, device,
                          fit_context):
    rng = random.Random(seed)
    seen = set()
    records = []
    attempts = 0
    while len(records) < n_docs:
        attempts += 1
        if attempts > 100*n_docs:
            raise RuntimeError('Too few score-only fit documents')
        plan = score_only_plan(rng, source='rotowire', group='train',
                               carrier_words=100, query_kind='winning_score')
        if plan.source_game_key in seen:
            continue
        text = score_only_realize(plan).text
        if fit_context == 'score_sentences_only':
            first_sentence = text.split('.', 1)[0]+'.'
            query = plan.query_form.replace('{SCORE}', str(plan.high))
            text = first_sentence+' '+query
        enc = tokenizer.encode(text,
                               add_special_tokens=False)
        length = len(enc.ids)-1
        if length > n_ctx or length < 10:
            continue
        seen.add(plan.source_game_key)
        records.append((enc.ids[:-1], length))
    covariance = torch.zeros((model.layers[2].d_head*model.layers[2].n_head,)*2,
                             dtype=torch.float64, device=device)
    positions = 0
    for start in range(0, len(records), 8):
        chunk = records[start:start+8]
        x = torch.full((len(chunk), n_ctx), PAD, dtype=torch.long, device=device)
        for i, (ids, length) in enumerate(chunk):
            x[i, :length] = torch.tensor(ids, device=device)
        h = model.embed(x)
        h = model.layers[0](h)
        h = model.layers[1](h)
        h = model.layers[2].norm(h)
        for i, (_, length) in enumerate(chunk):
            value = h[i, :length].double()
            covariance += value.T @ value
            positions += length
    covariance /= positions
    return covariance.cpu(), {'fit_docs': n_docs, 'fit_positions': positions,
                              'fit_context': fit_context,
                              'draw_attempts': attempts,
                              'fit_game_sha256': [hashlib.sha256(repr(key).encode()).hexdigest()
                                                  for key in sorted(seen)]}


@torch.no_grad()
def site_logits(model, batch, device):
    model.eval()
    outputs = []
    for variant in batch:
        parts = []
        for start in range(0, len(variant['x']), 8):
            end = min(start+8, len(variant['x']))
            x = variant['x'][start:end].to(device)
            sites = variant['site'][start:end].to(device)
            logits = model(x).float()
            ix = torch.arange(len(x), device=device)
            parts.append(torch.stack((logits[ix, sites[:, 0]],
                                      logits[ix, sites[:, 2]]), dim=1).cpu())
        outputs.append(torch.cat(parts))
    return outputs


def measure(reference, candidate, batch):
    n = len(batch[0]['x'])
    result = {}
    for j, kind in enumerate(KINDS):
        edit = 1 if kind == 'winning_score' else 2
        nuisance = 2 if kind == 'winning_score' else 1
        targets = [row['target'][:, 0 if j == 0 else 2]
                   for row in batch]
        correct = [candidate[i][:, j].argmax(-1) == targets[i]
                   for i in range(3)]
        probs = [candidate[i][:, j].softmax(-1) for i in range(3)]
        p_ref = [reference[i][:, j].softmax(-1) for i in range(3)]
        kl = sum((p_ref[i]*(reference[i][:, j].log_softmax(-1)-
                            candidate[i][:, j].log_softmax(-1))).sum(-1).sum().item()
                 for i in range(3))/(3*n)
        row = {'paired_top1': (correct[0] & correct[edit]).float().mean().item(),
               'all_variants_top1': sum(c.float().sum().item()
                                         for c in correct)/(3*n),
               'native_to_candidate_kl': kl,
               'targeted_full_tv': (.5*(probs[0]-probs[edit]).abs().sum(-1)).mean().item(),
               'nuisance_full_tv': (.5*(probs[0]-probs[nuisance]).abs().sum(-1)).mean().item()}
        old, new = targets[0], targets[edit]
        ix = torch.arange(n)
        before = candidate[0][:, j][ix, new]-candidate[0][:, j][ix, old]
        after = candidate[edit][:, j][ix, new]-candidate[edit][:, j][ix, old]
        row['edit_direction'] = (after > before).float().mean().item()
        result[kind] = row
    return result


def approximations(weight, covariance, ranks, seed):
    w = weight.double()
    c = covariance.double()
    d = len(c)
    regularizer = 1e-4*torch.trace(c)/d
    eigenvalues, eigenvectors = torch.linalg.eigh(c+regularizer*torch.eye(d))
    root = (eigenvectors*eigenvalues.sqrt()) @ eigenvectors.T
    invroot = (eigenvectors*eigenvalues.rsqrt()) @ eigenvectors.T
    u_w, s_w, v_w = torch.linalg.svd(w, full_matrices=False)
    u_d, s_d, v_d = torch.linalg.svd(w @ root, full_matrices=False)
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, max(ranks), generator=g,
                                      dtype=torch.float64))
    maps = {}
    for rank in ranks:
        maps[('weight_svd', rank)] = \
            (u_w[:, :rank]*s_w[:rank]) @ v_w[:rank]
        maps[('score_only_data_svd', rank)] = \
            ((u_d[:, :rank]*s_d[:rank]) @ v_d[:rank]) @ invroot
        maps[('random_input', rank)] = (w @ q[:, :rank]) @ q[:, :rank].T
    return maps, float(regularizer), {'min': eigenvalues.min().item(),
                                     'max': eigenvalues.max().item()}


@torch.no_grad()
def set_head_map(layer, head, desired):
    width = layer.d_head
    sl = slice(head*width, (head+1)*width)
    u, s, vh = torch.linalg.svd(desired.double(), full_matrices=False)
    rank = min(width, int((s > 1e-10).sum()))
    out = torch.zeros_like(layer.o.weight[:, sl])
    value = torch.zeros_like(layer.v.weight[sl])
    scale = s[:rank].sqrt()
    out[:, :rank] = (u[:, :rank]*scale).to(out.dtype)
    value[:rank] = (scale[:, None]*vh[:rank]).to(value.dtype)
    layer.o.weight[:, sl].copy_(out)
    layer.v.weight[sl].copy_(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--score-head', type=int, required=True)
    p.add_argument('--player-head', type=int, required=True)
    p.add_argument('--fit-docs', type=int, default=256)
    p.add_argument('--eval-docs', type=int, default=64)
    p.add_argument('--fit-seed', type=int, default=97001)
    p.add_argument('--fit-context', choices=('full_score_only',
                                             'score_sentences_only'),
                   default='full_score_only')
    p.add_argument('--eval-seed', type=int, default=97002)
    p.add_argument('--ranks', default='4,8,12,16')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    ranks = tuple(int(rank) for rank in a.ranks.split(','))
    if not ranks or max(ranks) > 16 or min(ranks) < 1:
        raise ValueError('Ranks must lie within the head width 1..16')
    model, config = load_model(a.checkpoint, a.device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    covariance, fit = score_only_covariance(model, tokenizer, a.fit_docs,
                                             a.fit_seed, config['n_ctx'], a.device,
                                             a.fit_context)
    records, attempts = draw(random.Random(a.eval_seed), a.eval_docs,
                              'heldout', tokenizer, config['n_ctx'], True,
                              carrier_words=config['carrier_words'])
    batch = stack(records)
    native_logits = site_logits(model, batch, a.device)
    native = measure(native_logits, native_logits, batch)
    layer = model.layers[2]
    original_out = layer.o.weight.detach().clone()
    original_value = layer.v.weight.detach().clone()
    cases = []
    role_maps = {}
    for role, head in (('score', a.score_head), ('player', a.player_head)):
        sl = slice(head*layer.d_head, (head+1)*layer.d_head)
        weight = original_out[:, sl] @ original_value[sl]
        maps, regularizer, eigens = approximations(
            weight.cpu(), covariance, ranks, a.fit_seed+head)
        role_maps[role] = maps
        for (method, rank), desired in maps.items():
            set_head_map(layer, head, desired.to(a.device))
            candidate_logits = site_logits(model, batch, a.device)
            metrics = measure(native_logits, candidate_logits, batch)
            error = (desired-weight.cpu().double())
            cases.append({'role': role, 'head': head, 'method': method,
                          'rank': rank,
                          'weight_relative_frobenius':
                              (error.norm()/weight.double().norm().cpu()).item(),
                          'fit_relative_output_rms':
                              torch.sqrt(torch.trace(error @ covariance @ error.T)/
                                         torch.trace(weight.cpu().double() @ covariance @
                                                     weight.cpu().double().T)).item(),
                          'metrics': metrics})
            with torch.no_grad():
                layer.o.weight.copy_(original_out)
                layer.v.weight.copy_(original_value)
        if role == 'score':
            score_regularizer, score_eigens = regularizer, eigens
    joint_cases = []
    for rank in ranks:
        for method in ('weight_svd', 'score_only_data_svd', 'random_input'):
            set_head_map(layer, a.score_head,
                         role_maps['score'][(method, rank)].to(a.device))
            set_head_map(layer, a.player_head,
                         role_maps['player'][(method, rank)].to(a.device))
            joint_cases.append({'method': method, 'rank': rank,
                                'metrics': measure(native_logits,
                                                   site_logits(model, batch, a.device),
                                                   batch)})
            with torch.no_grad():
                layer.o.weight.copy_(original_out)
                layer.v.weight.copy_(original_value)
    result = {'config': {**vars(a), 'checkpoint': str(a.checkpoint),
                         'output': str(a.output)},
              'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'fit': {**fit, 'covariance_regularizer': score_regularizer,
                      'regularized_covariance_eigenvalues': score_eigens},
              'eval': {'docs': a.eval_docs, 'draw_attempts': attempts,
                       'native': native}, 'cases': cases,
              'joint_cases': joint_cases,
              'caveat': 'Only one head OV product is reduced per case; QK and '
                        'other layers remain native. This is a local function '
                        'map, not full attention-tensor canonicalization. '
                        'Score-only activation fit excludes the inserted '
                        'player-copy task; full carrier may still contain '
                        'player-stat prose. score_sentences_only removes '
                        'that carrier for a strict rule-content exclusion. '
                        'Held-out evaluation includes both rules. '
                        'No model tensors or source prose saved.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for case in cases:
        if case['rank'] in (8, 16):
            print(case['role'], case['method'], case['rank'],
                  {k: round(v['paired_top1'], 3)
                   for k, v in case['metrics'].items()},
                  flush=True)


if __name__ == '__main__':
    main()
