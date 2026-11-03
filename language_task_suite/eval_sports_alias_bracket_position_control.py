"""Separate bracket-position shift from report-source transfer effects."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_alias_bracket_style_transfer import draw
from train_sports_alias_bracket import evaluate, stack
from train_sports_report_multirule import DeepModel, SPECS


SOURCE_DIR = Path('/home/loganriggs/Coding/neuron_circuits/'
                  'artifacts/rotowire')


def bucket(record):
    return min(4, int(record[1][0]['site'][1])//64)


def groups(records):
    out = {i: [] for i in range(5)}
    for record in records:
        out[bucket(record)].append(record)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=98351)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    states = [torch.load(path, map_location='cpu', weights_only=True)
              for path in a.checkpoints]
    configs = [state['config'] for state in states]
    assert len({c['n_ctx'] for c in configs}) == 1
    assert all(c['alias_prior'] == 'roster' and
               c['alias_exponent'] == 0 for c in configs)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records = {}
    counts = {}
    for source in ('rotowire', 'sbnation'):
        path = SOURCE_DIR/f'{source}.tar.bz2'
        records[source], counts[source] = draw(
            None, a.seed, tokenizer, configs[0]['n_ctx'], path)
    divided = {source: groups(rows) for source, rows in records.items()}
    matching = {i: min(len(divided[source][i]) for source in divided)
                for i in range(5)}
    matched = {source: [record for i in range(5)
                        for record in divided[source][i][:matching[i]]]
               for source in divided}
    assert len(matched['rotowire']) == len(matched['sbnation'])
    assert len(matched['rotowire']) >= 40
    results = []
    for path, state, cfg in zip(a.checkpoints, states, configs):
        model = DeepModel(50257, cfg['d_model'], cfg['heads'],
                          SPECS[f"block{cfg['depth']}"], cfg['n_ctx'],
                          norm='rms', residual='lerp',
                          attention='bilinear').to(a.device)
        model.load_state_dict(state['state_dict'])
        measured = {}
        for source in divided:
            measured[source] = {'all': evaluate(
                model, stack(records[source]), a.device),
                'matched': evaluate(model, stack(matched[source]), a.device),
                'bins': {str(i): evaluate(model, stack(rows), a.device)
                         for i, rows in divided[source].items() if rows}}
        results.append({'checkpoint_sha256': hashlib.sha256(
            path.read_bytes()).hexdigest(), 'metrics': measured})
        del model
    output = {'panel': {'seed': a.seed,
                        'source_counts': counts,
                        'bracket_position_bins': {
                            source: dict(Counter(bucket(row) for row in rows))
                            for source, rows in records.items()},
                        'matched_per_bin': matching,
                        'matched_n_per_source': len(matched['rotowire']),
                        'game_hashes': {source: [hashlib.sha256(
                            repr(key).encode()).hexdigest() for key, _ in rows]
                            for source, rows in records.items()}},
              'results': results,
              'caveat': ('Matched panels balance coarse bracket target '
                         'position bins, not exact position, source length, '
                         'lexicon, or game content. Bin-specific panels '
                         'are small. This remains a domain-transfer test '
                         'and surface coreference is heuristic.')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    print('Matched bins:', matching, flush=True)
    for i, result in enumerate(results):
        for source, measured in result['metrics'].items():
            print(i, source, 'all',
                  measured['all']['bracket']['paired_top1'], 'matched',
                  measured['matched']['bracket']['paired_top1'], flush=True)
    print('Saved', a.output)


if __name__ == '__main__':
    main()
