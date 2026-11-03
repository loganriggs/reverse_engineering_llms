"""Screen naturally occurring later winning-score mentions in RotoWire.

This is deliberately an opportunity screen. A numeric recurrence is not by
itself a semantically valid score-copy query or a safe counterfactual site.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_rotowire_document_source import game_key, score_pair_present
from corpus_score_sentence_renderer import _sentences
from sports_alias_bracket_dgp import bracket_pairs
from sports_document_dgp import game_partition, resources
from sports_report_carrier_dgp import _first_sentence_asserts_result


def audit():
    output = {group: Counter() for group in ('train', 'heldout')}
    manual = []
    for row in resources()[0]['rotowire'][0]:
        group = game_partition(game_key(row))
        if group not in output:
            continue
        count = output[group]
        try:
            home = int(row['home_line']['TEAM-PTS'])
            away = int(row['vis_line']['TEAM-PTS'])
        except (KeyError, ValueError):
            continue
        if home == away or not score_pair_present(row) or \
                not _first_sentence_asserts_result(row, home, away):
            continue
        count['validated_result_lead'] += 1
        high, low = max(home, away), min(home, away)
        sentences = list(_sentences(row['summary']))
        later = sentences[1:]
        has_high = [s for s in later if str(high) in s]
        has_no_low = [s for s in has_high if str(low) not in s]
        with_points = [s for s in has_no_low if any(
            token == str(high) and 'points' in
            (value.lower() for value in s[i+1:i+4])
            for i, token in enumerate(s))]
        bracket = bool(bracket_pairs(row['summary']))
        count['later_high_token'] += bool(has_high)
        count['later_high_without_low'] += bool(has_no_low)
        count['later_high_near_points'] += bool(with_points)
        count['later_high_near_points_with_bracket'] += \
            bool(with_points) and bracket
        if with_points and len(manual) < 16:
            manual.append((group, ' '.join(with_points[0])))
    return {group: dict(count) for group, count in output.items()}, manual


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    counts, manual = audit()
    result = {
        'groups': counts,
        'caveat': 'A later matching number can refer to the opponent, a '
                  'season record, or a part-of-total claim. No candidates '
                  'are automatically accepted as semantically valid or '
                  'editable. Raw source excerpts are printed only for '
                  'local manual inspection, never saved.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(counts, indent=2))
    print('Manual candidate excerpts (local only):')
    for group, sentence in manual:
        print(group, sentence)


if __name__ == '__main__':
    main()
