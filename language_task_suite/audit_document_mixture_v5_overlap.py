"""Fast exact-overlap audit for V5 record and score test documents.

Replicates the main RNG draws. V14 narrative uses only one seed draw from
that RNG; its internal token sampling is independent, so narrative text need
not be generated to preserve the record/score sequence.
"""

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from document_mixture_v5 import (KINDS, TASK_RICH_WEIGHTS, draw_record,
                                 draw_score, realize_record, realize_score)
from document_world_v4_interleaved import draw_interleaved


def draws(n, seed):
    rng = random.Random(seed)
    for _ in range(n):
        kind = rng.choices(KINDS, TASK_RICH_WEIGHTS)[0]
        if kind == 'narrative':
            rng.randrange(2**32)
            yield kind, None
        elif kind == 'event_report':
            draw_interleaved(rng)
            yield kind, None
        else:
            doc = (realize_record(draw_record(rng)) if kind == 'record_listing'
                   else realize_score(draw_score(rng)))
            digest = hashlib.blake2b('\0'.join(doc.tokens).encode(),
                                     digest_size=16).hexdigest()
            yield kind, digest


def audit(n_train, n_test, seed):
    train_counts = collections.Counter()
    seen = set()
    for kind, digest in draws(n_train, seed):
        train_counts[kind] += 1
        if digest:
            seen.add(digest)
    test_counts = collections.Counter()
    overlaps = collections.Counter()
    for kind, digest in draws(n_test, seed+1):
        test_counts[kind] += 1
        if digest is not None:
            overlaps[kind] += digest in seen
    return dict(n_train=n_train, n_test=n_test, seed=seed,
                train_kinds=dict(train_counts), test_kinds=dict(test_counts),
                exact_hash_overlap=dict(overlaps),
                scope='record and score exact documents only; narrative and event '
                      'generation skipped while preserving the main RNG',
                hash='blake2b-128 over null-separated token strings')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-train',type=int,default=20000)
    p.add_argument('--n-test',type=int,default=2000)
    p.add_argument('--seed',type=int,default=23012)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=audit(a.n_train,a.n_test,a.seed)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(result)
