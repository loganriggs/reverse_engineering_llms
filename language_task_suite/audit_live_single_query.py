"""Audit single-answer live ledgers and test frozen old-format models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import random

import torch

from eval_sports_live_ledger_edits import (
    edited, model_from_checkpoint, predictions)
from sports_live_ledger_dgp import (
    answer_from_visible_prefix, ledger_pool, sample_document)


ROOT = Path(__file__).resolve().parent


def generator_panel(n, seed, source, split):
    counts = Counter()
    for i in range(n):
        docs = [sample_document(random.Random(seed+i), source, split,
                                described, query_mode='single')
                for described in (False, True)]
        assert docs[0].latent['game_id'] == docs[1].latent['game_id']
        assert docs[0].latent['start_score'] == docs[1].latent['start_score']
        assert docs[0].latent['end_score'] == docs[1].latent['end_score']
        assert [(s.family, s.target) for s in docs[0].sites] == [
            (s.family, s.target) for s in docs[1].sites]
        counts['paired_descriptions'] += 1
        for doc in docs:
            assert len(doc.sites) == 1
            site = doc.sites[0]
            assert doc.tokens[site.index] == site.target
            kind = 'copy' if site.family == 'live_score_copy' else 'run'
            assert answer_from_visible_prefix(doc.tokens[:site.index],
                                              kind) == site.target
            earlier = doc.tokens[:site.index-3]
            assert not any(tuple(earlier[j:j+2]) in
                           (('Latest', ':'), ('Run', ':'))
                           for j in range(len(earlier)-1))
            counts[f'query/{kind}/{doc.latent["query_side"]}'] += 1
            counts['visible_sites'] += 1
            for side in (0, 1):
                offset = edited(doc, 'offset', side)
                assert int(offset.sites[0].target)-int(site.target) == (
                    int(kind == 'copy' and doc.latent['query_side'] == side))
                counts['offset_edits_checked'] += 1
                if not doc.latent['descriptions']:
                    play = edited(doc, 'add_play', side)
                    assert int(play.sites[0].target)-int(site.target) == (
                        int(doc.latent['query_side'] == side))
                    counts['play_edits_checked'] += 1
    return dict(counts)


@torch.no_grad()
def frozen_panel(model, vocab, n, seed, device):
    counts = defaultdict(Counter)
    for i in range(n):
        doc = sample_document(random.Random(seed+i), 'rotowire',
                              'heldout', False, query_mode='single')
        site = doc.sites[0]
        key = site.family
        before = predictions(model, vocab, doc, device)[key]
        label = f'{key}/side{doc.latent["query_side"]}'
        row = counts[label]
        row['n'] += 1
        row['correct'] += before == int(site.target)
        for edit, side in (('offset', 0), ('offset', 1),
                           ('add_play', 0), ('add_play', 1)):
            changed = edited(doc, edit, side)
            after = predictions(model, vocab, changed, device)[key]
            expected = (int(changed.sites[0].target)-int(site.target))
            item = counts[f'{label}/{edit}{side}/target_'
                          f'{"up" if expected else "same"}']
            item['n'] += 1
            item['response_exact'] += after-before == expected
            item['edited_correct'] += after == int(changed.sites[0].target)
            if edit == 'add_play':
                start_score = doc.latent['start_score']
                end_score = doc.latent['end_score']
                delta = [end_score[j]-start_score[j] for j in (0, 1)]
                delta[side] += 1
                supported = (min(delta) >= 1 and max(delta) >= 6 and
                             abs(delta[0]-delta[1]) >= 3)
                subset = counts[f'{label}/{edit}{side}/target_'
                                f'{"up" if expected else "same"}/'
                                f'{"in_support" if supported else "out_of_support"}']
                subset['n'] += 1
                subset['response_exact'] += after-before == expected
                subset['edited_correct'] += after == int(
                    changed.sites[0].target)
    return {key: dict(value) for key, value in counts.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--model-n', type=int, default=200)
    p.add_argument('--seed', type=int, default=52201)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--checkpoint', choices=(
        'old_control_depth4', 'old_paired_edits_depth4',
        'single_query_depth2', 'single_query_depth4',
        'single_query_9000_depth4'),
        help='Score only one checkpoint, useful on constrained devices')
    p.add_argument('--reuse-generator', type=Path,
                   help='Load an already audited generator panel and append '
                   'one model score without resampling')
    p.add_argument('--output', type=Path, default=ROOT / 'results/'
                   'live_single_query_audit_n200.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    pools = ledger_pool()
    for source in ('rotowire', 'sbnation'):
        assert not ({entry[0][0] for entry in pools[source, 'train']} &
                    {entry[0][0] for entry in pools[source, 'heldout']})
    out = {'config': {'n_per_source_split': a.n,
                      'frozen_model_heldout_n': a.model_n,
                      'seed': a.seed}, 'generator': {}, 'frozen_models': {},
           'caveat': 'Old-format checkpoints are evaluated out of '
                     'distribution; single-query checkpoints use their '
                     'own training distribution. Bare play edits are '
                     'coherent arithmetic counterfactuals but may leave '
                     'the window-selection support.'}
    if a.reuse_generator:
        out = json.loads(a.reuse_generator.read_text())
        assert out['config']['n_per_source_split'] == a.n
        assert out['config']['frozen_model_heldout_n'] == a.model_n
        assert out['config']['seed'] == a.seed
        assert len(out['generator']) == 4
    else:
        for source in ('rotowire', 'sbnation'):
            for split in ('train', 'heldout'):
                key = f'{source}/{split}'
                value = generator_panel(a.n,
                                        a.seed+1000*len(out['generator']),
                                        source, split)
                out['generator'][key] = value
                print(key, value, flush=True)
    checkpoints = {
        'old_control_depth4': ROOT / 'results/'
            'sports_live_ledger_bare_3000_control_bare_depth4.pt',
        'old_paired_edits_depth4': ROOT / 'results/'
            'sports_live_ledger_bare_3000_paired_edits_bare_depth4.pt',
        'single_query_depth2': ROOT / 'results/'
            'sports_live_ledger_single_query_3000_bare_depth2.pt',
        'single_query_depth4': ROOT / 'results/'
            'sports_live_ledger_single_query_3000_bare_depth4.pt',
        'single_query_9000_depth4': ROOT / 'results/'
            'sports_live_ledger_single_query_9000_bare_depth4.pt',
    }
    selected = ({a.checkpoint: checkpoints[a.checkpoint]}
                if a.checkpoint else {})
    for name, path in selected.items():
        model, vocab, _ = model_from_checkpoint(path, a.device)
        result = frozen_panel(model, vocab, a.model_n, a.seed+5000,
                              a.device)
        out['frozen_models'][name] = result
        print(name, {k: v for k, v in result.items() if '/' not in k},
              flush=True)
        del model
        gc.collect()
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(out, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
