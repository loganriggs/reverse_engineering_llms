"""Score frozen bilinear checkpoints on a fresh whole-combination panel.

Only aggregate numbers are written. The FineWeb token panel and checkpoints
stay local because the former can reconstruct source text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval_sports_ov_canonical_truncation import model_from_checkpoint
from train_three_source_rule_mixture import joint_evaluate


ROOT = Path(__file__).resolve().parent
DEFAULT_PANEL = ROOT / 'results/joint_style_delimiter_fresh_84500_94500.pt'
DEFAULT_STYLE = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/'
                     'corpus_spelling_style_pilot/dataset.pt')
DEFAULT_DELIMITER = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/'
                         'corpus_delimiter_pilot/dataset.pt')


def chunk_set(rows, width=64, stride=16):
    chunks = set()
    for row in rows:
        values = row.tolist()
        for start in range(0, len(values)-width+1, stride):
            part = values[start:start+width]
            if 50256 not in part:
                chunks.add(tuple(part))
    return chunks


def overlap_screen(panel, style_cache, delimiter_cache):
    training = set()
    for path in (style_cache, delimiter_cache):
        cache = torch.load(path, map_location='cpu', weights_only=True)
        training.update(chunk_set(cache['train']['x']))
    joint = chunk_set(panel['arms']['us', 'round']['x'])
    return {'training_unique_64_token_chunks': len(training),
            'joint_unique_64_token_chunks': len(joint),
            'joint_chunks_matching_training': len(joint & training),
            'scope': 'exact 64-token chunks at stride 16; not a near-duplicate audit'}


def checkpoint_paths():
    for seed, sports_seed in ((41701, 98201), (41702, 98202)):
        yield str(seed), {
            'sports_start': ROOT / 'results' / (
                f'sports_alias_bracket_depth2_seed{sports_seed}_6000.pt'),
            'sports_spelling': ROOT / 'results' / (
                f'three_source_rule_mixture_seed{seed}_1000_sports_style.pt'),
            'three_source': ROOT / 'results' / (
                f'three_source_rule_mixture_seed{seed}_1000_three_source.pt')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', type=Path, default=DEFAULT_PANEL)
    parser.add_argument('--style-cache', type=Path, default=DEFAULT_STYLE)
    parser.add_argument('--delimiter-cache', type=Path, default=DEFAULT_DELIMITER)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/'
                        'three_source_joint_fresh_84500_94500.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    panel = torch.load(args.panel, map_location='cpu', weights_only=True)
    assert panel['counts']['aligned'] == len(panel['arms']['us', 'round']['x'])
    results = {'panel': panel['config'], 'counts': panel['counts'],
               'form_first100': panel['form_first100'],
               'exact_chunk_overlap': overlap_screen(
                   panel, args.style_cache, args.delimiter_cache),
               'frozen_checkpoint_scores': {}}
    for seed, paths in checkpoint_paths():
        results['frozen_checkpoint_scores'][seed] = {}
        for label, path in paths.items():
            model, _ = model_from_checkpoint(path, args.device)
            score = joint_evaluate(model, panel, args.device,
                                   batch_size=args.batch_size)
            results['frozen_checkpoint_scores'][seed][label] = score
            print(seed, label, score, flush=True)
            del model
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
