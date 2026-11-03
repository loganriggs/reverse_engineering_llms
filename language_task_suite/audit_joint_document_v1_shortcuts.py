"""Measure source-answer shortcuts on untouched joint-DGP streams."""

import argparse
import collections
import json
from pathlib import Path

from joint_document_v1 import _split, sample_sequence, visible_state


def audit(n, length, seed):
    counts = collections.Counter()
    for row in range(n):
        words, labels, _ = sample_sequence(seed+row, length)
        for i, label in enumerate(labels):
            if label != 'source_answer':
                continue
            history, raw = _split(words[:i])
            state = visible_state(history)
            key = (raw[3], raw[5])
            claims = list(state['claims'].items())
            answer = state['claims'][key]
            speakers = {speaker for _, speaker in claims}
            item_speakers = {speaker for (item, _), speaker in claims
                             if item == key[0]}
            counts['n_source_answers'] += 1
            counts['last_claim_speaker_correct'] += (answer == claims[-1][1])
            counts['query_key_is_latest_claim'] += (key == claims[-1][0])
            counts['multiple_speakers_visible'] += (len(speakers) >= 2)
            counts['item_has_unique_speaker'] += (len(item_speakers) == 1)
            counts['item_has_multiple_speakers'] += (len(item_speakers) >= 2)
            if len(speakers) >= 2:
                counts['last_speaker_correct_given_multiple'] += (
                    answer == claims[-1][1])
    n_sites = counts['n_source_answers']
    return dict(version='joint_document_v1', n_streams=n,
                length=length, seed=seed, counts=dict(counts),
                rates={key: value/n_sites for key, value in counts.items()
                       if key != 'n_source_answers'},
                caveat='Baselines use only visible claims; no trained-model '
                       'scores or causal edits. Independent stream seeds '
                       'are not whole-task holdouts.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--seed', type=int, default=126001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.n, args.length, args.seed)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['rates'], flush=True)
