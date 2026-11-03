"""Paired semantic and form screen for coordinated typed introductions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_document_world_v5_clause_realizations import _edited_world
from audit_surface_repetition import WORD
from compare_v12_fineweb_fresh import first_words
from document_world_v10_fact_acts import (
    answer_from_visible_prefix as answer_v10,
    draw_varied as draw_v10, realize as realize_v10)
from document_world_v13_packed_intro import (
    _split_intro_sentence, answer_from_visible_prefix,
    realize as realize_v13)


def _sentences(tokens):
    current = []
    for token in tokens:
        if token == '.':
            yield current
            current = []
        elif token != '\n':
            current.append(token)


def _intro_facts(doc):
    found = {}
    for sentence in _sentences(doc.tokens):
        if sentence[:1] in (['Story'], ['Report'], ['File']):
            continue
        try:
            rows = _split_intro_sentence(sentence)
        except (AssertionError, ValueError):
            break
        for obj, state, alias, prep in rows:
            assert obj not in found
            found[obj] = (state, alias, prep)
    mapping = doc.latent['surface_lexicon']
    expected = {mapping.get(obj, obj): (
        {key: mapping.get(value, value) for key, value in state.items()},
        doc.latent['aliases'][obj],
        'on' if state['place'] in ('shelf', 'desk') else 'in')
        for obj, state in doc.latent['initial'].items()}
    assert found == expected, (found, expected)
    return len(found)


def _sites(doc, lexical, candidate):
    counts = Counter()
    for site in doc.sites:
        assert doc.tokens[site.index] == site.answer
        if candidate:
            observed = answer_from_visible_prefix(
                doc.tokens[:site.index], lexical_enabled=lexical)
        else:
            observed = answer_v10(
                doc.tokens[:site.index], lexical_enabled=lexical)
        assert observed == site.answer, (site.family, observed, site.answer)
        counts[site.family] += 1
    return counts


def _window(doc, site, width):
    words = list(WORD.finditer(doc.text()))
    target = len(WORD.findall(' '.join(doc.tokens[:site.index])))
    if len(words) < width:
        return None
    start = max(0, min(target-width//2, len(words)-width))
    return doc.text()[words[start].start():words[start+width-1].end()]


def panel(seed, n, width, words, pack_rates):
    rng = random.Random(seed)
    forms = {name: [] for name in ('v10',) + tuple(
        f'pack_{rate:g}' for rate in pack_rates)}
    site_windows = {name: defaultdict(list) for name in forms}
    counts = Counter()
    examples = []
    while len(forms['v10']) < n:
        base = draw_v10(rng, lexical_exponent=0., act_rate=.8,
                        syntax_entropy=1., aside_rate=.3,
                        context_rate=.35, source_delay_rate=.5,
                        extra_source_recap_rate=.5)
        world = base.latent
        arms = {'v10': realize_v10(world, lexical_enabled=True,
                                  syntax_entropy=1., aside_rate=.3,
                                  context_rate=.35)}
        for rate in pack_rates:
            arms[f'pack_{rate:g}'] = realize_v13(
                world, pack_rate=rate, lexical_enabled=True,
                syntax_entropy=1., aside_rate=.3, context_rate=.35)
        counts['draws'] += 1
        crops = {name: first_words(doc.text(), words)
                 for name, doc in arms.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        counts['eligible'] += 1
        for name, doc in arms.items():
            candidate = name != 'v10'
            if candidate:
                counts['parsed_intro_facts'] += _intro_facts(doc)
            counts.update({'site/'+name+'/'+family: count for family, count
                           in _sites(doc, True, candidate).items()})
            forms[name].append(crops[name])
            for family in set(site.family for site in doc.sites):
                site = next(s for s in doc.sites if s.family == family)
                crop = _window(doc, site, width)
                if crop:
                    site_windows[name][family].append(crop)
        # Check a supported upstream edit on the most packed candidate.
        plan = world['recap_plan'][0]
        changed = _edited_world(world, plan)
        altered = realize_v13(changed, pack_rate=max(pack_rates),
                              lexical_enabled=True, syntax_entropy=1.,
                              aside_rate=.3, context_rate=.35)
        _intro_facts(altered)
        _sites(altered, True, True)
        original = arms[f'pack_{max(pack_rates):g}']
        assert altered.sites[0].answer != original.sites[0].answer
        counts['supported_edit_checks'] += 1
        if len(examples) < 3:
            examples.append({name: doc.text()[:450] for name, doc in
                             arms.items()})
        assert counts['draws'] <= 200*n
    return {'counts': dict(counts),
            'first100_form': {name: structure(rows)
                              for name, rows in forms.items()},
            'site_window_form': {name: {family: structure(rows)
                                       for family, rows in groups.items()}
                                 for name, groups in site_windows.items()},
            'site_window_counts': {name: {family: len(rows)
                                         for family, rows in groups.items()}
                                   for name, groups in site_windows.items()},
            'examples': examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--window-words', type=int, default=41)
    p.add_argument('--seed', type=int, default=53701)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rates = (0., .75, 1.)
    panels = [panel(a.seed+i, a.n, a.window_words, a.words, rates)
              for i in range(2)]
    result = {'config': {'n': a.n, 'words': a.words,
                         'window_words': a.window_words,
                         'seeds': [a.seed, a.seed+1], 'pack_rates': rates},
              'panels': panels,
              'caveat': 'Paired sampled worlds; eligibility requires all '
                        'arms to have at least 100 words. Source prose is '
                        'not used. This is a semantic and form screen, '
                        'not model competence or a full-token oracle.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for i, row in enumerate(panels):
        print('seed', a.seed+i, 'draws', row['counts']['draws'])
        for name, form in row['first100_form'].items():
            print(name, form['word_types'],
                  round(form['top10_word_share'], 3),
                  round(form['repeated_across_docs_trigram_occurrence_share'], 3),
                  flush=True)


if __name__ == '__main__':
    main()
