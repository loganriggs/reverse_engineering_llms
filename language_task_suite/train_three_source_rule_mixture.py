"""Train separate natural-text rule sources, test unseen joint combination.

The joint FineWeb style×delimiter panel is never used for fitting. All
source arrays and model checkpoints remain local; output JSON is aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_sports_alias_bracket import draw, evaluate as sports_evaluate, stack
from train_sports_report_multirule import DeepModel, PAD, SPECS
from train_corpus_spelling_style_bilinear import _pair_metrics
from train_corpus_delimiter_bilinear import evaluate as delimiter_evaluate
from finetune_joint_natural_rules import weighted_loss, indexed


def pad_rows(rows, n_ctx):
    width = rows['x'].shape[1]
    if width > n_ctx:
        raise ValueError('Source window exceeds model context')
    out = dict(rows)
    out['x'] = F.pad(rows['x'], (0, n_ctx-width), value=PAD)
    out['y'] = F.pad(rows['y'], (0, n_ctx-width), value=-100)
    return out


@torch.no_grad()
def joint_evaluate(model, panel, device, batch_size=2):
    model.eval()
    arms = panel['arms']
    n = len(arms['us', 'round']['x'])
    totals = dict(style_pair=0, delimiter_pair=0,
                  style_direction=0, delimiter_direction=0,
                  all_four_two_way=0)
    for start in range(0, n, batch_size):
        end = min(n, start+batch_size)
        rows = {arm: indexed(data, slice(start, end), device)
                for arm, data in arms.items()}
        logits = {arm: model(row['x']).float()
                  for arm, row in rows.items()}
        m = end-start
        ix = torch.arange(m, device=device)
        site = {arm: [logits[arm][ix, rows[arm]['site'][:, kind]]
                      for kind in (0, 1)] for arm in rows}
        style_ok, delimiter_ok = [], []
        for bracket in ('round', 'square'):
            u, k = ('us', bracket), ('uk', bracket)
            us_id = rows[u]['target'][:, 0]
            uk_id = rows[k]['target'][:, 0]
            good_u = site[u][0][ix, us_id] > site[u][0][ix, uk_id]
            good_k = site[k][0][ix, uk_id] > site[k][0][ix, us_id]
            style_ok.append(good_u & good_k)
            shift = ((site[u][0][ix, us_id]-site[u][0][ix, uk_id])-
                     (site[k][0][ix, us_id]-site[k][0][ix, uk_id]))
            totals['style_direction'] += (shift > 0).sum().item()
        for style in ('us', 'uk'):
            r, s = (style, 'round'), (style, 'square')
            round_id = rows[r]['target'][:, 1]
            square_id = rows[s]['target'][:, 1]
            good_r = site[r][1][ix, round_id] > site[r][1][ix, square_id]
            good_s = site[s][1][ix, square_id] > site[s][1][ix, round_id]
            delimiter_ok.append(good_r & good_s)
            shift = ((site[r][1][ix, round_id]-site[r][1][ix, square_id])-
                     (site[s][1][ix, round_id]-site[s][1][ix, square_id]))
            totals['delimiter_direction'] += (shift > 0).sum().item()
        totals['style_pair'] += sum(row.sum().item() for row in style_ok)
        totals['delimiter_pair'] += sum(row.sum().item()
                                        for row in delimiter_ok)
        totals['all_four_two_way'] += (style_ok[0] & style_ok[1] &
                                       delimiter_ok[0] &
                                       delimiter_ok[1]).sum().item()
    return {'documents': n,
            'style_paired_two_way': totals['style_pair']/(2*n),
            'delimiter_paired_two_way': totals['delimiter_pair']/(2*n),
            'style_edit_direction': totals['style_direction']/(2*n),
            'delimiter_edit_direction': totals['delimiter_direction']/(2*n),
            'both_rules_all_four_arms': totals['all_four_two_way']/n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-checkpoint', type=Path, required=True)
    p.add_argument('--style-cache', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_spelling_style_pilot/dataset.pt'))
    p.add_argument('--delimiter-cache', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_delimiter_pilot/dataset.pt'))
    p.add_argument('--joint-panel', type=Path, required=True)
    p.add_argument('--train-worlds', type=int, default=1000)
    p.add_argument('--test-worlds', type=int, default=40)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--seed', type=int, default=41701)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.source_checkpoint, map_location='cpu',
                       weights_only=True)
    cfg = saved['config']
    if cfg['depth'] != 2 or cfg['n_ctx'] != 320:
        raise ValueError('Expected depth-2 context-320 sports checkpoint')
    style = torch.load(args.style_cache, map_location='cpu',
                       weights_only=True)
    delimiter = torch.load(args.delimiter_cache, map_location='cpu',
                           weights_only=True)
    panel = torch.load(args.joint_panel, map_location='cpu',
                       weights_only=True)
    assert style['vocab_size'] == delimiter['vocab_size'] == 50257
    assert panel['config']['n_ctx'] == cfg['n_ctx']
    style_train = pad_rows(style['train'], cfg['n_ctx'])
    style_test = {key: pad_rows(value, cfg['n_ctx'])
                  for key, value in style['test'].items()}
    delimiter_train = pad_rows(delimiter['train'], cfg['n_ctx'])
    delimiter_test = {key: pad_rows(value, cfg['n_ctx'])
                      for key, value in delimiter['test'].items()}
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(args.seed)
    sports_train, train_attempts = draw(
        rng, args.train_worlds, 'train', tokenizer, cfg['n_ctx'])
    sports_test, test_attempts = draw(
        rng, args.test_worlds, 'heldout', tokenizer, cfg['n_ctx'],
        unique=True)
    assert not {key for key, _ in sports_train} & \
        {key for key, _ in sports_test}
    sports_train, sports_test = stack(sports_train), stack(sports_test)
    flat_sports = {key: torch.cat([arm[key] for arm in sports_train])
                   for key in ('x', 'y', 'site', 'target')}

    def create_model():
        model = DeepModel(50257, cfg['d_model'], cfg['heads'],
                          SPECS['block2'], cfg['n_ctx'], norm='rms',
                          residual='lerp', attention='bilinear').to(args.device)
        model.load_state_dict(saved['state_dict'])
        return model

    def score(model):
        return {'sports': sports_evaluate(model, sports_test, args.device,
                                          batch_size=4),
                'style': _pair_metrics(model, style_test, args.device,
                                       batch_size=2),
                'delimiter': delimiter_evaluate(model, delimiter_test,
                                                 args.device, batch_size=2),
                'joint': joint_evaluate(model, panel, args.device)}

    result = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
              'data': {'sports_train_worlds': args.train_worlds,
                       'sports_test_worlds': args.test_worlds,
                       'sports_train_attempts': train_attempts,
                       'sports_test_attempts': test_attempts,
                       'style_train': len(style_train['x']),
                       'delimiter_train': len(delimiter_train['x']),
                       'joint_holdout_documents': panel['counts']['aligned'],
                       'joint_source_hashes': panel['source_sha256']},
              'objective': 'Controlled source mixture: sports alias+bracket, '
                           'FineWeb spelling only, FineWeb delimiter only. '
                           'The whole FineWeb style×delimiter combination '
                           'is withheld. Each source has a normalized '
                           'weighted token loss; sports weights 500/100, '
                           'spelling and delimiter weights 100.',
              'starting': score(create_model()), 'arms': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for arm in ('sports_style', 'three_source'):
        torch.manual_seed(args.seed)
        model = create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sample = random.Random(args.seed+100)
        losses = []
        started = time.time()
        for step in range(args.steps):
            model.train()
            counts = (4, 4, 0) if arm == 'sports_style' else (3, 3, 2)
            sources = (flat_sports, style_train, delimiter_train)
            weights = ((500., 100.), (100.,), (100.,))
            terms = []
            optimizer.zero_grad(set_to_none=True)
            for count, source, task_weights in zip(counts, sources, weights):
                if not count:
                    continue
                indices = torch.tensor([sample.randrange(len(source['x']))
                                        for _ in range(count)])
                batch = {key: value[indices].to(args.device)
                         for key, value in source.items()}
                terms.append(weighted_loss(model, batch, task_weights))
            loss = sum(terms)/len(terms)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite {arm} step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
            if (step+1) % 250 == 0:
                print(arm, step+1, round(loss.item(), 4), flush=True)
        result['arms'][arm] = {
            'last50_weighted_loss': sum(losses[-50:])/len(losses[-50:]),
            'seconds': round(time.time()-started, 1),
            'test': score(model)}
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        torch.save({'state_dict': {key: value.detach().cpu()
                                   for key, value in model.state_dict().items()},
                    'config': cfg, 'arm': arm},
                   args.output.with_name(args.output.stem+f'_{arm}.pt'))
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
