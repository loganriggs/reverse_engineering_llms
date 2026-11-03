"""Measure V9's sampled lexical-prior concentration, separately from prose."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from document_world import NAMES, OBJECTS
from document_world_v9_typed_lexicon import draw_lexicon


def run(n, seed, exponent):
    rng = random.Random(seed)
    counts = {'people': Counter(), 'objects': Counter()}
    for _ in range(n):
        mapping = draw_lexicon(rng, exponent)
        counts['people'].update(mapping[name] for name in NAMES)
        counts['objects'].update(mapping[obj] for obj in OBJECTS)
    return {category: dict(
        types=len(c), top10_share=sum(value for _, value in c.most_common(10))
        / sum(c.values())) for category, c in counts.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--seed', type=int, default=77511)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(config=dict(n=a.n, seed=a.seed),
                  exponents={str(e): run(a.n, a.seed, e)
                             for e in (0., .5, 1.)},
                  caveat='Measures selected lexicon slots, not token '
                         'frequency in complete documents.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    main()
