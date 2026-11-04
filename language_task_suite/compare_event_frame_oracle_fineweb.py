"""Fresh 40-word form screen for variable versus fixed event frames."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import sys

from datasets import load_dataset

from audit_surface_repetition import WORD, form_cues, score
from event_frame_oracle import EventFrameOracle


def first_n_words(text, n):
    matches = list(islice(WORD.finditer(text), n))
    return text[:matches[-1].end()] if len(matches) == n else None


def metrics(rows):
    words = [word.lower() for row in rows for word in WORD.findall(row)]
    freq = Counter(words)
    starts = Counter()
    for row in rows:
        for segment in row.replace('?', '.').replace('!', '.').split('.'):
            found = WORD.findall(segment)
            if found:
                starts[found[0].lower()] += 1
    return {'n': len(rows), 'types': len(freq),
            'top_ten_fraction': sum(v for _, v in freq.most_common(10))/len(words),
            'repeated_word_fraction': score(rows, 40)['repeated_word_fraction'],
            'first_segment_start_types': len({next(iter(WORD.findall(row)), '')
                                              for row in rows}),
            'all_segment_start_types': len(starts),
            'top_five_segment_start_fraction': sum(v for _, v in
                  starts.most_common(5))/sum(starts.values()),
            'forms': form_cues(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=440000)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--seed', type=int, default=42011)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    try:
        natural = [row['text'] for row in islice(stream, a.start,
                                                a.start+a.n)]
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert len(natural) == a.n
    variable = EventFrameOracle(seed=a.seed, variable_frames=True)
    fixed = EventFrameOracle(seed=a.seed, variable_frames=False)
    left, right = [], []
    for _ in range(a.n):
        v, f = variable.sample_document(), fixed.sample_document()
        assert v.events == f.events and v.genre == f.genre
        left.append(variable.render(v.tokens))
        right.append(fixed.render(f.tokens))
    raw = {'fineweb': natural, 'variable': left, 'fixed': right}
    clipped = {name: [first_n_words(text, 40) for text in rows]
               for name, rows in raw.items()}
    eligible = [i for i in range(a.n)
                if all(rows[i] is not None for rows in clipped.values())]
    if not eligible:
        raise RuntimeError('No common 40-word panel')
    result = {'config': {**vars(a), 'output': str(a.output)},
              'source_documents': a.n,
              'jointly_eligible_40_word_documents': len(eligible),
              'series': {name: metrics([rows[i] for i in eligible])
                         for name, rows in clipped.items()},
              'scope': 'Fresh unselected FineWeb source offsets and paired '
                       'latent synthetic events. All comparisons use the '
                       'first 40 alphabetic words of each eligible text. '
                       'A 40-word screen is a form diagnostic; grammar '
                       'validity and mechanism coverage require separate '
                       'checks. No FineWeb source prose saved.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, row in result['series'].items():
        print(name, row['types'], row['top_ten_fraction'],
              row['repeated_word_fraction'], row['all_segment_start_types'],
              flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
