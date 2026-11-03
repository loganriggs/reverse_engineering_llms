"""Lower bound on V10's trigram reuse under slot-word changes alone.

The mask is deliberately broad: every typed lexicon value, alias, topic,
incidental noun/adjective/event is treated as a variable slot. Repeated
trigrams outside this mask cannot be changed by replacing slot words while
leaving the sentence scaffold fixed. This is a *sample-specific* lower
bound, not an estimate of what a new grammar can achieve.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from audit_surface_repetition import WORD
from compare_v12_fineweb_fresh import first_words
from document_world_v3_multisite import ALIASES
from document_world_v5_clause_realizations import INCIDENTAL_NOUNS
from document_world_v8_discourse import draw_varied as draw_v8
from document_world_v9_typed_lexicon import draw_lexicon
from document_world_v10_fact_acts import plan_acts, realize
from surface_mixture_v2 import ADJECTIVES, EVENTS


def audit(seed, n=64, words=100):
    rng = random.Random(seed)
    rows = []
    draws = 0
    while len(rows) < n:
        draws += 1
        world = draw_v8(rng, syntax_entropy=1.,
                        source_delay_rate=0.).latent
        world['fact_act_plan'] = plan_acts(world, .8)
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019), 0.)
        text = first_words(realize(
            world, lexical_enabled=True, syntax_entropy=1.).text(), words)
        tokens = [word.lower() for word in WORD.findall(text)]
        if len(tokens) < words:
            continue
        variable = ({word.lower() for word in
                     world['surface_lexicon'].values()} |
                    {word.lower() for word in ALIASES} |
                    {world['topic'].lower()} |
                    {word.lower() for word in INCIDENTAL_NOUNS} |
                    {word.lower() for word in ADJECTIVES} |
                    {word.lower() for word in EVENTS})
        rows.append([(tuple(tokens[i:i+3]),
                      any(token in variable for token in tokens[i:i+3]))
                     for i in range(len(tokens)-2)])
    doc_frequency = Counter(gram for row in rows
                            for gram in {gram for gram, _ in row})
    counts = Counter(gram for row in rows for gram, _ in row)
    total = sum(map(len, rows))
    summary = {}
    for name, predicate in (
            ('all', lambda variable: True),
            ('touches_variable', lambda variable: variable),
            ('frame_only', lambda variable: not variable)):
        selected = [gram for row in rows for gram, variable in row
                    if predicate(variable)]
        repeated = sum(doc_frequency[gram] > 1 for gram in selected)
        summary[name] = dict(occurrences=len(selected), repeated=repeated,
                             fraction_of_all=repeated/total,
                             within_class=repeated/len(selected))
    fixed_counts = Counter(gram for row in rows for gram, variable in row
                           if not variable)
    summary['common_frame_trigrams'] = [
        [' '.join(gram), count] for gram, count in fixed_counts.most_common(15)]
    return dict(seed=seed, n=n, words=words, draws=draws,
                summary=summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(panels=[audit(94121), audit(94122)],
                  caveat='Conservative slot mask on generated text only. '
                         'No natural text bodies or model behavior.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for panel in result['panels']:
        print(panel['seed'], panel['summary']['all'],
              panel['summary']['frame_only'])


if __name__ == '__main__':
    main()
