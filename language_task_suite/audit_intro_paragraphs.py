"""Visible-rule and paired form screen for sampled intro paragraphs."""

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
from document_world_intro_paragraphs import (
    answer_from_visible_prefix, draw_varied, plan, realize,
)
from document_world_v10_fact_acts import plan_acts, realize as acts_realize
from document_world_v8_discourse import draw_varied as draw_v8
from document_world_v9_typed_lexicon import draw_lexicon


def semantic(seed=2917, n=300):
    rng = random.Random(seed)
    result = Counter()
    for _ in range(n):
        base = draw_v8(rng, syntax_entropy=1., source_delay_rate=1.,
                       extra_source_recap_rate=.4)
        world = base.latent
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019), 0.)
        for lexical in (False, True):
            doc = realize(world, syntax_entropy=1.,
                          lexical_enabled=lexical)
            for site in doc.sites:
                assert answer_from_visible_prefix(
                    doc.tokens[:site.index], lexical) == site.answer
                result['scored_sites'] += 1
        # Read each initial fact from only the new introduction text.
        intro = [word for sentence in plan(world) for word in sentence]
        for obj, state in world['initial'].items():
            for attr, value in state.items():
                query = (['Current', attr if attr != 'place' else 'location',
                          'of', obj, ':'])
                assert answer_from_visible_prefix(intro + query) == value
                result['initial_fact_checks'] += 1
        obj = rng.choice(tuple(world['initial']))
        attr = rng.choice(tuple(VALUES))
        edited = copy.deepcopy(world)
        old = edited['initial'][obj][attr]
        replacement = rng.choice([v for v in VALUES[attr] if v != old])
        edited['initial'][obj][attr] = replacement
        intro = [word for sentence in plan(edited) for word in sentence]
        query = ['Current', attr if attr != 'place' else 'location',
                 'of', obj, ':']
        assert answer_from_visible_prefix(intro + query) == replacement
        result['targeted_initial_fact_edits'] += 1
    return dict(result)


def form(seed, n=64, words=100):
    rng = random.Random(seed)
    texts = {key: [] for key in (
        'prior_best', 'intro_default', 'intro_no_aside',
        'intro_acts_no_aside')}
    draws = 0
    while len(texts['prior_best']) < n:
        draws += 1
        world = draw_v8(rng, syntax_entropy=1.,
                        source_delay_rate=0.).latent
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019), 0.)
        world['fact_act_plan'] = plan_acts(world, .8)
        docs = dict(
            prior_best=acts_realize(
                world, syntax_entropy=1., lexical_enabled=True),
            intro_default=realize(
                world, syntax_entropy=1., lexical_enabled=True),
            intro_no_aside=realize(
                world, syntax_entropy=1., lexical_enabled=True,
                aside_rate=0.),
            intro_acts_no_aside=acts_realize(
                world, syntax_entropy=1., lexical_enabled=True,
                aside_rate=0., initial_sentence_plan=plan(world)))
        crops = {key: first_words(doc.text(), words)
                 for key, doc in docs.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        for key in texts:
            texts[key].append(crops[key])
    return dict(seed=seed, draws=draws,
                panels={key: structure(rows) for key, rows in texts.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(semantic=semantic(), form=[form(94121), form(94122)],
                  caveat='Paired 100-word form and visible-text semantic '
                         'checks only; no trained model or full-token '
                         'conditional oracle.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['semantic'])
    for panel in result['form']:
        for name, row in panel['panels'].items():
            print(panel['seed'], name, row['word_types'],
                  round(row['repeated_across_docs_trigram_occurrence_share'],
                        3), row['sentence_start_types'])


if __name__ == '__main__':
    main()
