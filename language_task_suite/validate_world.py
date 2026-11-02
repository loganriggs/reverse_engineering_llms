"""Oracle, split, and paired-edit checks for document-level tasks."""

import argparse
import importlib
from collections import Counter

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world-version', choices=('v1', 'v2'), default='v1')
    args = p.parse_args()
    world = importlib.import_module('document_world' if args.world_version == 'v1'
                                    else 'document_world_v2')
    GENRES, causal_pair, composition_fold, nuisance_pair, sample_world, world_oracle = (
        getattr(world, name) for name in ('GENRES', 'causal_pair', 'composition_fold',
                                          'nuisance_pair', 'sample_world', 'world_oracle'))
    train = sample_world(20_000, 42, genres=(0, 1))
    iid = sample_world(4_000, 43, genres=(0, 1))
    held_compositions = sample_world(4_000, 44, split='composition', genres=(0, 1))
    held_genre = sample_world(4_000, 45, genres=(2,))
    seen = {}
    for split in (train, iid, held_compositions, held_genre):
        for ex in split:
            assert ex.target == world_oracle(ex), ex
            assert ex.target in ex.candidates
            assert seen.setdefault(ex.prefix, ex.target) == ex.target
            assert (composition_fold(ex) == 0) == (split is held_compositions)
    for ex in train[:5000] + held_compositions[:5000] + held_genre[:5000]:
        a, b = causal_pair(ex)
        assert a.target != b.target
        assert b.target == world_oracle(b)
        a, b = nuisance_pair(ex)
        assert a.target == b.target
        assert b.target == world_oracle(b)
    train_prefixes = {ex.prefix for ex in train}
    overlap = {name: sum(ex.prefix in train_prefixes for ex in split)
               for name, split in [('iid', iid), ('composition', held_compositions),
                                   ('genre', held_genre)]}
    assert overlap == {'iid': 0, 'composition': 0, 'genre': 0}, overlap
    if args.world_version == 'v2':
        train_vocab = {token for ex in train for token in ex.prefix + (ex.target,)}
        unseen = {token for ex in held_genre for token in ex.prefix + (ex.target,)
                  if token not in train_vocab}
        assert not unseen, ('unseen held-genre tokens', unseen)
        assert all('a orange' not in ex.text() for split in
                   (train, iid, held_compositions, held_genre) for ex in split)
    print('PASS', sum(map(len, (train, iid, held_compositions, held_genre))),
          'examples, oracle and pairs; no train/test exact-prefix overlap')
    print('lengths', {name: (min(map(lambda x: len(x.prefix), split)),
                             max(map(lambda x: len(x.prefix), split)))
                      for name, split in [('train', train), ('iid', iid),
                                          ('composition', held_compositions),
                                          ('genre', held_genre)]})
    print('families', dict(Counter(ex.family for ex in train)))
    print('genres', GENRES)


if __name__ == '__main__':
    main()
