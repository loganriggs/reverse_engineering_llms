"""Fresh game-disjoint confirmation panel for multistyle fine-tuning."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_rotowire_document_source import game_key, load
from eval_sports_alias_bracket_style_transfer import draw
from train_sports_alias_bracket import evaluate, stack
from train_sports_report_multirule import DeepModel, SPECS


SOURCE_DIR = Path('/home/loganriggs/Coding/neuron_circuits/'
                  'artifacts/rotowire')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--originals', nargs=2, type=Path, required=True)
    p.add_argument('--mixed', nargs=2, type=Path, required=True)
    p.add_argument('--controls', nargs=2, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=98451)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    paths = {'original': a.originals, 'mixed': a.mixed,
             'rotowire_control': a.controls}
    states = {arm: [torch.load(path, map_location='cpu', weights_only=True)
                    for path in pair] for arm, pair in paths.items()}
    configs = [state['config'] for rows in states.values() for state in rows]
    if len({(c['n_ctx'], c['d_model'], c['heads'], c['depth'])
            for c in configs}) != 1:
        raise ValueError('Incompatible model architectures')
    cfg = configs[0]
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    panels, counts = {}, {}
    for source in ('rotowire', 'sbnation'):
        archive = SOURCE_DIR/f'{source}.tar.bz2'
        source_rows = load(archive, source)
        exclude = {game_key(row) for row in source_rows['train']}
        records, count = draw(None, a.seed, tokenizer, cfg['n_ctx'],
                              archive, group='heldout',
                              splits=('valid', 'test'), excluded=exclude)
        assert all(key not in exclude for key, _ in records)
        panels[source] = stack(records)
        counts[source] = {
            **count, 'n': len(records),
            'bracket_position_bins': dict(Counter(
                min(4, int(rows[0]['site'][1])//64)
                for _, rows in records)),
            'game_hashes': [hashlib.sha256(repr(key).encode()).hexdigest()
                            for key, _ in records],
        }
    results = {}
    for arm, rows in states.items():
        results[arm] = []
        for path, state in zip(paths[arm], rows):
            model = DeepModel(50257, cfg['d_model'], cfg['heads'],
                              SPECS[f"block{cfg['depth']}"], cfg['n_ctx'],
                              norm='rms', residual='lerp',
                              attention='bilinear').to(a.device)
            model.load_state_dict(state['state_dict'])
            results[arm].append({
                'checkpoint_sha256': hashlib.sha256(
                    path.read_bytes()).hexdigest(),
                'metrics': {source: evaluate(model, batch, a.device)
                            for source, batch in panels.items()}})
            del model
    result = {
        'panel': {'seed': a.seed, 'sources': counts,
                  'split': 'source valid/test, held-out game partition, '
                           'exclude all games in each source train archive'},
        'results': results,
        'caveat': ('SBNation fresh panel has only 15 games. It is independent '
                   'by game from the training and periodically inspected '
                   'source-train held-out panels, but is too small for a '
                   'precise transfer estimate. Human coreference labels '
                   'remain surface-heuristic.')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for arm, rows in results.items():
        print(arm, [round(row['metrics']['sbnation']['bracket'][
            'paired_top1'], 3) for row in rows], flush=True)
    print('Saved', a.output)


if __name__ == '__main__':
    main()
