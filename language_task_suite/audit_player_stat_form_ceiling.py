"""Count how much natural player-stat prose a points-only plan can license."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import PLAYER, _detokenize, resources


POINT_MENTION = re.compile(r'\b(\d+) points\b')
STRICT = re.compile(
    r'^([A-Z][a-z]+ [A-Z][a-z]+) '
    r'(scored|posted|finished with|had|added|contributed|tallied|'
    r'registered|put up) (\d+) points'
    r'(?: (in the (?:win|loss|game)|on the night))?\.$')


def screen(rows):
    counts = Counter()
    strict_forms = Counter()
    for row in rows:
        box = row['box_score']
        known = {(name, box['PTS'].get(key))
                 for key, name in box['PLAYER_NAME'].items()
                 if PLAYER.fullmatch(name)}
        for sentence in _sentences(row['summary']):
            counts['sentences'] += 1
            text = _detokenize(sentence)
            mentions = {(name, points)
                        for name, points in known
                        if points and points.isdigit() and
                        re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)', text)
                        and points in POINT_MENTION.findall(text)}
            if not mentions:
                continue
            counts['matched_player_points'] += 1
            if len(mentions) != 1:
                counts['multiple_matched_player_points'] += 1
                continue
            counts['one_matched_player_points'] += 1
            if len([token for token in sentence if token.isdigit()]) > 1:
                counts['other_numeric_tokens'] += 1
            else:
                counts['one_numeric_token'] += 1
            match = STRICT.fullmatch(text)
            if match and (match[1], match[3]) in mentions:
                counts['strict_points_only_sentence'] += 1
                strict_forms[(match[2], match[4] or '')] += 1
    return {'counts': dict(counts),
            'strict_forms': [dict(verb=verb, qualifier=qualifier,
                                  count=count)
                             for (verb, qualifier), count in
                             strict_forms.most_common()]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    output = {name: screen(rows)
              for name, (rows, _) in resources()[0].items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    for name, result in output.items():
        print(name, result['counts'])


if __name__ == '__main__':
    main()
