"""Aggregate syntax of human numeric-run clauses in joined sports reports.

Screens surface structure, not truth or complete semantic entailment. No
human source passage is written to the output artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

from audit_sports_report_run_mentions import RUN
from corpus_score_sentence_renderer import _sentences
from sports_event_natural_carrier import eligible_event_carriers


PAIR = re.compile(r'\b\d+\s*-\s*\d+\b')
NUMBER = re.compile(r'\b\d+\b')
CLAUSE = re.compile(r'\b(?:but|and|as|after|before|while|which|who|'
                    r'because|although|despite|with|when|where|then|'
                    r'however|eventually|finally)\b', re.I)


def panel(source, group):
    counts = Counter()
    lengths = []
    positions = []
    before_run = Counter()
    after_run = Counter()
    for _, row, entry in eligible_event_carriers(source, group, 100):
        counts['eligible_reports'] += 1
        sentences = list(_sentences(row['summary']))
        for i, sentence in enumerate(sentences):
            text = ' '.join(sentence)
            for match in RUN.finditer(text):
                x, y = map(int, match.groups())
                if not 5 <= x <= 40 or not 0 <= y <= 30:
                    continue
                counts['numeric_run_mentions'] += 1
                lengths.append(len(sentence))
                position = len(text[:match.start()].split())/max(1, len(sentence))
                positions.append(position)
                counts['run_after_first_third'] += position >= 1/3
                counts['run_after_half'] += position >= .5
                counts['has_clause_cue'] += bool(CLAUSE.search(text))
                counts['has_comma_or_semicolon'] += ',' in sentence or \
                    ';' in sentence
                counts['has_other_score_pair'] += len(PAIR.findall(text)) > 1
                rest = text[:match.start()]+text[match.end():]
                counts['has_other_number'] += bool(NUMBER.search(rest))
                prior = (' '.join(' '.join(previous) for previous in
                                  sentences[max(0, i-3):i])+' '+
                         text[:match.start()])
                prior_pairs = PAIR.findall(prior)
                counts['at_least_one_prior_score_pair'] += \
                    len(prior_pairs) >= 1
                counts['at_least_two_prior_score_pairs'] += \
                    len(prior_pairs) >= 2
                counts['short_sentence_le16_tokens'] += len(sentence) <= 16
                home, away = entry[1:3]
                counts['names_full_team'] += home in text or away in text
                counts['names_either_nickname'] += any(
                    team.split()[-1] in sentence for team in (home, away))
                if i and PAIR.search(' '.join(sentences[i-1])):
                    counts['previous_sentence_has_score_pair'] += 1
                before = text[:match.start()].lower().split()
                after = text[match.end():].lower().split()
                if before:
                    before_run[' '.join(before[-2:])] += 1
                if after:
                    after_run[' '.join(after[:2])] += 1
    return {'counts': dict(counts),
            'mean_sentence_tokens': round(statistics.mean(lengths), 2)
            if lengths else None,
            'median_sentence_tokens': statistics.median(lengths)
            if lengths else None,
            'mean_run_position_fraction': round(statistics.mean(positions), 3)
            if positions else None,
            'top_two_words_before_run': before_run.most_common(12),
            'top_two_words_after_run': after_run.most_common(12)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'scope': 'Event-linked 2016/17 human reports with visible final '
                    'score and >=100 source tokens; whole-game split',
           'caveat': 'Lexical and punctuation proxies overlap. Other '
                     'numbers or a matching score pair do not establish '
                     'a distinct factual clause or causal entailment.',
           'arms': {}}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            key = source+'_'+group
            out['arms'][key] = panel(source, group)
            print(key, out['arms'][key], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
