"""Check exact oracles, source contrasts and supported edits in V2."""

import argparse
import collections
import json
from pathlib import Path

from joint_document_v1 import _split, site_label, visible_state
from joint_document_v2 import next_distribution, sample_sequence


def supported(prefix, regime):
    return all(next_distribution(prefix[:i], regime).get(token, 0.) > 0
               for i, token in enumerate(prefix))


def attribute_edit(prefix, regime):
    history, raw = _split(prefix)
    assert raw[:2] == ('Who', 'described') and len(raw) == 9
    claims = visible_state(history)['claims']
    current = (raw[3], raw[5])
    gold = claims[current]
    choices = [attr for (item, attr), speaker in claims.items()
               if item == current[0] and speaker != gold]
    for attr in choices:
        edited = list(prefix)
        edited[-4] = attr
        if supported(edited, regime):
            return edited
    return None


def audit(regime, n, length, seed):
    sites, edits, shortcut = (collections.Counter() for _ in range(3))
    lexical = collections.Counter()
    all_five = 0
    for row in range(n):
        words, labels, distributions = sample_sequence(seed+row, length, regime)
        seen = set()
        for i, (word, label, recorded) in enumerate(zip(
                words, labels, distributions, strict=True)):
            prefix = words[:i]
            q = next_distribution(prefix, regime)
            assert q == recorded and label == site_label(prefix)
            assert q.get(word, 0) > 0 and abs(sum(q.values())-1)<1e-10
            sites[label] += 1
            seen.add(label)
            if label == 'source_answer':
                history, raw = _split(prefix)
                claims = visible_state(history)['claims']
                key = (raw[3], raw[5])
                gold = claims[key]
                assert q == {gold: 1.}
                shortcut['last_claim_correct'] += (gold == list(claims.values())[-1])
                shortcut['item_multi_speaker'] += len({speaker for (item, _), speaker
                                                       in claims.items() if item == key[0]}) >= 2
                edited = attribute_edit(prefix, regime)
                if edited:
                    new = next_distribution(edited, regime)
                    assert len(new) == 1 and next(iter(new)) != gold
                    edits['attribute_key_changes_answer'] += 1
            elif label == 'bracket_close':
                _, raw = _split(prefix)
                opener = next(token for token in raw if token in ('(', '['))
                changed = list(prefix)
                changed[len(prefix)-len(raw)+raw.index(opener)] = (
                    '[' if opener == '(' else '(')
                assert supported(changed, regime)
                assert next(iter(q)) != next(iter(next_distribution(changed, regime)))
                edits['opener_changes_closer'] += 1
            elif label == 'section_number':
                history, _ = _split(prefix)
                expected = str(1+sum(s[0] in ('Section', 'Part') for s in history))
                assert q == {expected: 1.}
                edits['section_replayed'] += 1
        all_five += {'mixed_verb', 'claim_linked_item', 'source_answer',
                     'section_number', 'bracket_close'} <= seen
        if row < 49:
            alpha = [w.lower() for w in words if w.isalpha()]
            assert len(alpha) >= 100
            lexical.update(alpha[:100])
    assert sites['source_answer'] > 0
    return dict(version='joint_document_v2', regime=regime, seed=seed,
                n_streams=n, length=length, oracle_checks=n*length,
                sites=dict(sites), paired_edits=dict(edits),
                all_five_families_streams=all_five,
                source_shortcuts=dict(shortcut),
                source_shortcut_rates={k: v/sites['source_answer']
                                       for k, v in shortcut.items()},
                first49_x100_alpha=dict(types=len(lexical),
                                        singletons=sum(v == 1 for v in lexical.values()),
                                        top10_share=sum(v for _, v in lexical.most_common(10))/4900),
                caveat='Exact structural DGP audit; the FineWeb surface gate '
                       'is separate and remains unmet.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--regime', choices=('reference', 'task_rich'), required=True)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--seed', type=int, default=26001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.regime, args.n, args.length, args.seed)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({key: result[key] for key in ('sites', 'paired_edits',
                                       'source_shortcut_rates',
                                       'first49_x100_alpha')}, flush=True)
