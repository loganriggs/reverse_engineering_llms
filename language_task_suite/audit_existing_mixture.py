"""Audit the existing autoregressive mixed-rule generators before adapting them."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

DEFAULT_REPO = '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=os.environ.get('ATTN_CIRCUITS_DIR', DEFAULT_REPO))
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--output', default='language_task_suite/existing_mixture_audit.json')
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.source).resolve()))
    from generator import LanguageGenerator, ALL_RULES, DEFAULT_MIXING
    from generator_v3 import LanguageV3

    old = LanguageGenerator(ALL_RULES, mixing_weights=DEFAULT_MIXING,
                            mode='mixed', seed=173)
    new = LanguageV3(seed=173)
    old_counts, new_counts = Counter(), Counter()
    active_counts, collisions = Counter(), 0
    mixed_positions = 0
    for _ in range(a.n):
        ids, labels = old.sample_sequence(a.length)
        toks = [old.id2token[i] for i in ids]
        old_counts.update(labels)
        for t in range(1, len(toks)):
            if not labels[t].startswith(('bigram_', 'trigram_', 'skip_', 'induction_')):
                continue
            context = toks[max(0, t-old.window_size):t]
            candidates = {}
            for cls, fn in [('bigram', old._try_bigram),
                            ('trigram', old._try_trigram),
                            ('skip_bigram', old._try_skip_bigram),
                            ('induction', old._try_induction)]:
                candidate, _ = fn(context)
                if candidate is not None:
                    candidates[cls] = candidate
            mixed_positions += 1
            active_counts[len(candidates)] += 1
            collisions += len(set(candidates.values())) > 1 and len(candidates) > 1
        _, labels = new.sample_sequence(a.length)
        new_counts.update(labels)

    # A controlled collision: the last 'a' activates bigram a→X0 and
    # induction from the earlier a→B0. No other ordinary rule is active.
    collision_context = ['a', 'B0', 'a']
    draws = Counter(old._sample_next_token(collision_context)[0]
                    for _ in range(20_000))
    expected_x0 = DEFAULT_MIXING['bigram'] / (
        DEFAULT_MIXING['bigram'] + DEFAULT_MIXING['induction'])
    observed_x0 = draws['X0'] / sum(draws.values())
    assert set(draws) == {'X0', 'B0'}, draws
    assert abs(observed_x0 - expected_x0) < 0.02, (observed_x0, expected_x0)

    total = a.n*a.length
    result = dict(config=vars(a),
                  old=dict(mixing_weights=DEFAULT_MIXING,
                           label_counts=dict(old_counts),
                           rule_positions=mixed_positions,
                           active_candidate_counts=dict(active_counts),
                           conflicting_output_fraction_of_rule_positions=collisions/max(1,mixed_positions),
                           controlled_collision=dict(context=collision_context,
                               outcomes=dict(draws), expected_p_X0=expected_x0,
                               observed_p_X0=observed_x0)),
                  v3=dict(label_counts=dict(new_counts)),
                  caveat='The original generator chooses among active candidates after '
                         'normalizing their configured weights; setup and bracket tokens '
                         'are separate paths. V3 uses priority dispatch, not an unconditional '
                         'mixture over all rule classes. Labels describe sampled paths, '
                         'not a complete conditional-probability oracle.')
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print('v1 conflicting active outputs', round(result['old'][
          'conflicting_output_fraction_of_rule_positions'], 4))
    print('v1 candidate counts', dict(active_counts))
    print('controlled bigram/induction collision',
          round(observed_x0, 4), 'expected', round(expected_x0, 4))
    print('v3 labels', dict(new_counts))


if __name__ == '__main__':
    main()
