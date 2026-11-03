"""Reproducible semantic and paired-form audit for composed event clauses."""

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from audit_surface_repetition import WORD
from compare_v12_fineweb_fresh import first_words
from document_world import VALUES
from document_world_composed_events import (
    answer_from_visible_prefix, plan, realize,
)
from document_world_v8_discourse import (
    draw_varied as draw_v8, realize as realize_v8,
)
from document_world_v9_typed_lexicon import draw_lexicon, lexicalize_document


def semantic(seed, n):
    rng = random.Random(seed)
    result = Counter()
    for _ in range(n):
        world = draw_v8(rng, source_delay_rate=1.,
                        extra_source_recap_rate=.4).latent
        world['composed_event_plan'] = plan(world, .8)
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019), 0.)
        for kind in ('value', 'source'):
            options = []
            for site_index, (after, family, obj, attr) in enumerate(
                    world['recap_plan']):
                query_attr = family.removeprefix('world_')
                if ((kind == 'value' and
                     query_attr not in VALUES) or
                        (kind == 'source' and
                         family != 'source_attribution')):
                    continue
                if kind == 'value':
                    attr = query_attr
                relevant = [j for j in range(after+1)
                            if (world['events'][j]['object'],
                                world['events'][j]['attribute']) ==
                            (obj, attr)]
                if relevant:
                    options.append((site_index, relevant[-1], attr))
            if not options:
                continue
            site_index, event_index, attr = rng.choice(options)
            original = realize(world)
            edited = copy.deepcopy(world)
            if kind == 'value':
                old = original.sites[site_index].answer
                edited['events'][event_index]['value'] = rng.choice(
                    [v for v in VALUES[attr] if v != old])
            else:
                old = original.sites[site_index].answer
                edited['sources'][event_index] = rng.choice(
                    [v for v in VALUES['owner'] if v != old])
            target = realize(edited).sites[site_index]
            assert target.answer != old
            assert answer_from_visible_prefix(
                realize(edited).tokens[:target.index]) == target.answer
            result['targeted_' + kind + '_flips'] += 1
        for lexical in (False, True):
            doc = realize(world, lexical_enabled=lexical)
            for site in doc.sites:
                assert answer_from_visible_prefix(
                    doc.tokens[:site.index], lexical) == site.answer
                result['original_sites'] += 1
            # Change an actual prior event, then re-render the side claims.
            edited = copy.deepcopy(world)
            j = rng.randrange(len(world['events']))
            attr = edited['events'][j]['attribute']
            old = edited['events'][j]['value']
            edited['events'][j]['value'] = rng.choice(
                [v for v in VALUES[attr] if v != old])
            changed = realize(edited, lexical_enabled=lexical)
            assert len(changed.sites) == len(doc.sites)
            for before, after in zip(doc.sites, changed.sites, strict=True):
                assert answer_from_visible_prefix(
                    changed.tokens[:after.index], lexical) == after.answer
                result['edited_sites'] += 1
                result['changed_answers' if before.answer != after.answer
                       else 'stable_answers'] += 1
            result['value_edits'] += 1
            if not lexical:
                # Corrupt an explicitly asserted pre-update fact while
                # keeping the event unchanged. The visible checker must
                # reject the inconsistency, rather than silently strip it.
                for index, token in enumerate(doc.tokens):
                    if token != 'While' or index+7 >= len(doc.tokens):
                        continue
                    later = next((s for s in doc.sites if s.index > index+8),
                                 None)
                    if later is None:
                        continue
                    attr = doc.tokens[index+7]
                    old = doc.tokens[index+4]
                    if attr not in VALUES or old not in VALUES[attr]:
                        continue
                    altered = list(doc.tokens)
                    altered[index+4] = next(
                        v for v in VALUES[attr] if v != old)
                    try:
                        answer_from_visible_prefix(
                            altered[:later.index], False)
                    except AssertionError:
                        result['false_side_claims_rejected'] += 1
                    else:
                        raise AssertionError('False side claim accepted')
                    break
    return dict(result)


def form(seed, n=64, words=100):
    rng = random.Random(seed)
    texts = dict(base=[], composed=[])
    draws = 0
    while len(texts['base']) < n:
        draws += 1
        world = draw_v8(rng, syntax_entropy=1.,
                        source_delay_rate=0.,
                        extra_source_recap_rate=0.).latent
        world['composed_event_plan'] = plan(world, .8)
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019), 0.)
        docs = dict(
            base=lexicalize_document(
                realize_v8(world, syntax_entropy=1.), world),
            composed=realize(world, syntax_entropy=1.,
                             lexical_enabled=True))
        crops = {k: first_words(doc.text(), words)
                 for k, doc in docs.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        for key in texts:
            texts[key].append(crops[key])
    frequent = {}
    for key, rows in texts.items():
        grams = Counter()
        for text in rows:
            tokens = [w.lower() for w in WORD.findall(text)]
            grams.update(tuple(tokens[i:i+3])
                         for i in range(len(tokens)-2))
        frequent[key] = [(' '.join(gram), count)
                         for gram, count in grams.most_common(12)]
    return dict(seed=seed, draws=draws,
                panels={key: structure(rows) for key, rows in texts.items()},
                frequent_trigrams=frequent)


def form_positions(seed=499, n=64):
    rng = random.Random(seed)
    texts = {key: {'first': [], 'last': []}
             for key in ('base', 'composed')}
    draws = 0
    while len(texts['base']['first']) < n:
        draws += 1
        world = draw_v8(rng, syntax_entropy=1.).latent
        world['composed_event_plan'] = plan(world, .8)
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019), 0.)
        docs = dict(
            base=lexicalize_document(
                realize_v8(world, syntax_entropy=1.), world),
            composed=realize(world, syntax_entropy=1.,
                             lexical_enabled=True))
        words = {key: WORD.findall(doc.text()) for key, doc in docs.items()}
        if min(len(row) for row in words.values()) < 120:
            continue
        for key, row in words.items():
            texts[key]['first'].append(' '.join(row[:60]))
            texts[key]['last'].append(' '.join(row[-60:]))
    return dict(seed=seed, draws=draws,
                panels={key: {position: structure(rows)
                              for position, rows in by_position.items()}
                        for key, by_position in texts.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(semantic=semantic(77192, 300),
                  form=[form(94121), form(94122)],
                  position_form=form_positions(),
                  caveat='First-100-word paired form screen and visible '
                         'semantic checks; no trained model or exact '
                         'full-token oracle.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['semantic'])
    for panel in result['form']:
        for name, row in panel['panels'].items():
            print(panel['seed'], name, row['word_types'],
                  row['repeated_across_docs_trigram_occurrence_share'],
                  row['sentence_start_types'])


if __name__ == '__main__':
    main()
