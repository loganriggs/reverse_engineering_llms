"""Count natural carrier games eligible for score, alias and bracket rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sports_alias_bracket_dgp import bracket_pairs
from sports_entity_alias_dgp import _rows, candidates
from sports_report_carrier_dgp import eligible as score_eligible


def audit(words, group):
    original = _rows()
    counts = {'score': 0, 'score_and_alias': 0,
              'score_and_bracket': 0, 'all_three': 0}
    for key, *middle, carrier, home_win in score_eligible(
            'rotowire', group, words=words):
        counts['score'] += 1
        row = dict(original[key])
        row['summary'] = list(carrier)
        alias = bool(candidates(row))
        bracket = bool(bracket_pairs(carrier))
        counts['score_and_alias'] += alias
        counts['score_and_bracket'] += bracket
        counts['all_three'] += alias and bracket
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--words', default='50,100,150,200')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {
        'by_carrier_words': {str(words): {
            group: audit(words, group)
            for group in ('train', 'heldout')}
            for words in map(int, a.words.split(','))},
        'interpretation': 'Counts unique game keys after conservative '
                          'score-semantic screen and after removing score '
                          'sentences from the natural carrier. Bracket '
                          'pairs must remain in that carrier; surname '
                          'source/target must also remain. This screens '
                          'opportunity, not BPE alignment or model learning.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['by_carrier_words'], indent=2))


if __name__ == '__main__':
    main()
