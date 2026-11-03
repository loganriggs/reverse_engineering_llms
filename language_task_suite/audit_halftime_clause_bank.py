"""Conservative full-sentence halftime fact template screen.

Only aggregate counts and hashed normalized forms are written. The source
clauses stay in memory; this script does not certify arbitrary game prose.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from audit_rotowire_document_source import game_key
from audit_halftime_rule_opportunities import (
    MARKERS, half_pair_positions, quarter_values,
)
from sports_document_dgp import game_partition, resources


PAIR = r'(?P<a>\d+) - (?P<b>\d+)'
HALF = r'(?:at (?:the )?(?:half|halftime|intermission)|by halftime|into (?:the )?half)'
PATTERNS = {
    'lead_direct': re.compile(
        rf'^(?:The )?TEAM (?:led|lead|leads|was up|were up|is up|are up|'
        rf'leading) {PAIR} {HALF} \.$', re.I),
    'lead_after_time': re.compile(
        rf'^(?:The )?TEAM (?:led|lead|leads) (?:at the half|at halftime) '
        rf'{PAIR} \.$', re.I),
    'lead_noun': re.compile(
        rf'^(?:The )?TEAM (?:took|had|carried|held) a {PAIR} '
        rf'(?:lead|advantage) {HALF} \.$', re.I),
    'lead_goes': re.compile(
        rf'^(?:The )?TEAM (?:went|goes) into halftime up {PAIR} \.$',
        re.I),
    'tie': re.compile(
        rf'^(?:The )?game was tied (?:at (?:the )?half|at halftime) , '
        rf'{PAIR} \.$', re.I),
}


def source_sentences(tokens):
    start = 0
    for i, token in enumerate(tokens):
        if token == '.':
            yield start, i+1, tuple(tokens[start:i+1])
            start = i+1


def team_mentions(tokens, row):
    text = ' '.join(tokens)
    found = []
    for side in ('home', 'vis'):
        aliases = (row[side+'_city']+' '+row[side+'_name'],
                   row[side+'_city'], row[side+'_name'])
        matches = []
        for alias in aliases:
            pattern = re.compile(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',
                                 re.I)
            matches.extend((m.start(), m.end(), alias)
                           for m in pattern.finditer(text))
        if matches:
            found.append((side, max(matches, key=lambda x: x[1]-x[0])))
    return text, found


def accepted(tokens, row, halftime):
    if len(tokens) > 25 or tokens[0] == '*NEWPARAGRAPH*':
        return None
    pair_positions = half_pair_positions(tokens, halftime)
    if len(pair_positions) != 1:
        return None
    if sum(token.isdigit() for token in tokens) != 2:
        return None
    if any(token in ('?', '!', ';', '``', "''") for token in tokens):
        return None
    text, mentions = team_mentions(tokens, row)
    if len(mentions) > 1:
        return None
    if mentions:
        side, (_, _, alias) = mentions[0]
        # Normalize the *longest* matched team name, then require the entire
        # resulting sentence to fit one supported grammar production.
        pattern = re.compile(r'(?<!\w)'+re.escape(alias)+r'(?!\w)', re.I)
        normalized, count = pattern.subn('TEAM', text)
        if count != 1:
            return None
    else:
        normalized, side = text, None
    for kind, pattern in PATTERNS.items():
        match = pattern.fullmatch(normalized)
        if match is None or (kind == 'tie') != (side is None):
            continue
        a, b = int(match['a']), int(match['b'])
        if (a, b) not in (halftime, halftime[::-1]):
            continue
        if kind == 'tie':
            if halftime[0] != halftime[1]:
                continue
            orientation = 'tie'
        else:
            team_score = halftime[0 if side == 'home' else 1]
            other_score = halftime[1 if side == 'home' else 0]
            if not kind.startswith('lead') or team_score <= other_score:
                continue
            # In these lead constructions the first score belongs to the
            # named leader. Merely matching the two values in either order
            # would accept contradictory text ("Boston led 58-59"). Trail
            # constructions vary in score order, so this narrow grammar
            # rejects them until orientation can be validated separately.
            if (a, b) != (team_score, other_score):
                continue
            orientation = 'team_first'
        skeleton = re.sub(r'\b\d+ - \d+\b', 'SCORE - SCORE', normalized)
        return {'kind': kind, 'orientation': orientation,
                'skeleton': skeleton, 'team_side': side}
    return None


def audit_row(row):
    values = quarter_values(row)
    if values is None:
        return None
    halftime, _ = values
    found = []
    for _, _, sentence in source_sentences(row['summary']):
        if not any(token.lower() in MARKERS for token in sentence):
            continue
        if not half_pair_positions(sentence, halftime):
            continue
        match = accepted(sentence, row, halftime)
        if match:
            found.append(match)
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            counts = Counter()
            forms = Counter()
            for row in resources()[0][source][0]:
                if game_partition(game_key(row)) != group:
                    continue
                counts['documents'] += 1
                accepted_rows = audit_row(row)
                if accepted_rows is None:
                    continue
                counts['quarter_consistent'] += 1
                counts['accepted_documents'] += bool(accepted_rows)
                counts['accepted_clauses'] += len(accepted_rows)
                for match in accepted_rows:
                    counts['kind/'+match['kind']] += 1
                    counts['orientation/'+match['orientation']] += 1
                    forms[match['skeleton']] += 1
            result[source+'/'+group] = {
                'counts': dict(counts),
                'distinct_normalized_forms': len(forms),
                'hashed_form_counts': {
                    hashlib.sha256(form.encode()).hexdigest()[:16]: n
                    for form, n in forms.items()},
            }
            print(source, group, dict(counts), 'forms', len(forms),
                  flush=True)
    result['caveat'] = ('This is a deliberately narrow full-sentence grammar. '
                        'Acceptance certifies only the listed score/team/'
                        'lead relation, under the parser assumptions. It '
                        'does not establish a natural-rate DGP or a safe '
                        'intervention on the entire original report. '
                        'Source text and unhashed templates are not saved.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
