"""Semantic and form gates for the three-purpose DGP pilot.

Uses paired worlds across purposes, supported value/source edits, and stored
FineWeb aggregate references. No natural document bodies are persisted.
"""

import argparse
import copy
import json
import random
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from audit_surface_repetition import WORD
from compare_v12_fineweb_fresh import first_words
from document_world import VALUES
from document_world_purpose_pilot import (
    PURPOSES, answer_from_visible_prefix, realize,
)
from document_world_v8_discourse import draw_varied as draw_v8
from document_world_v9_typed_lexicon import draw_lexicon, lexicalize_document


def check(doc):
    for site in doc.sites:
        assert answer_from_visible_prefix(doc.tokens[:site.index]) == site.answer
    return len(doc.sites)


def audit(seed, n, words):
    rng = random.Random(seed)
    texts = {p: [] for p in PURPOSES}
    lexical_texts = {p: [] for p in PURPOSES}
    counts = dict(worlds=0, visible_sites=0, value_edits=0,
                  source_edits=0, changed_answers=0,
                  unchanged_answers=0)
    while counts['worlds'] < n:
        base = draw_v8(rng, source_delay_rate=1.,
                       extra_source_recap_rate=.4)
        world = base.latent
        docs = {p: realize(world, p) for p in PURPOSES}
        crops = {p: first_words(d.text(), words) for p, d in docs.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        counts['worlds'] += 1
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019), 0.)
        for p, doc in docs.items():
            counts['visible_sites'] += check(doc)
            texts[p].append(crops[p])
            lexical_texts[p].append(first_words(
                lexicalize_document(doc, world).text(), words))
        for kind in ('value', 'source'):
            edited = copy.deepcopy(world)
            j = rng.randrange(len(world['events']))
            if kind == 'value':
                attr = world['events'][j]['attribute']
                current = world['events'][j]['value']
                candidates = [v for v in VALUES[attr] if v != current]
                edited['events'][j]['value'] = rng.choice(candidates)
                counts['value_edits'] += 1
            else:
                current = world['sources'][j]
                candidates = [v for v in VALUES['owner'] if v != current]
                edited['sources'][j] = rng.choice(candidates)
                counts['source_edits'] += 1
            for p in PURPOSES:
                before, after = docs[p], realize(edited, p)
                assert len(before.sites) == len(after.sites)
                for old, new in zip(before.sites, after.sites, strict=True):
                    assert (old.family, old.object) == (new.family, new.object)
                    assert answer_from_visible_prefix(
                        after.tokens[:new.index]) == new.answer
                    key = ('changed_answers' if old.answer != new.answer
                           else 'unchanged_answers')
                    counts[key] += 1
    return dict(seed=seed, n=n, words=words, counts=counts,
                structural={p: structure(texts[p]) for p in PURPOSES},
                lexical_structural={p: structure(lexical_texts[p])
                                    for p in PURPOSES})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=94121)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--words', type=int, default=100)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.seed, args.n, args.words)
    reference = json.loads(args.reference.read_text())
    result['reference'] = dict(
        path=str(args.reference),
        fineweb=reference['panels']['fineweb'],
        prior_best=reference['panels']['both'])
    result['caveat'] = ('Paired first-100-word form screen and supported '
                        'semantic edits; neither is a model-behavior result '
                        'nor an exact full-token probability oracle.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('semantic', result['counts'])
    for label, rows in [('natural', {'fineweb': result['reference']['fineweb'],
                                    'prior_best': result['reference']['prior_best']}),
                        ('purpose', result['structural']),
                        ('lexical', result['lexical_structural'])]:
        for name, row in rows.items():
            print(label, name, 'types', row['word_types'], 'trigrams',
                  round(row['repeated_across_docs_trigram_occurrence_share'],
                        3), 'starts', row['sentence_start_types'])


if __name__ == '__main__':
    main()
