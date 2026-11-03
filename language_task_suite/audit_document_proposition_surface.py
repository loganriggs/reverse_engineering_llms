"""Visible semantic, counterfactual, and natural-form gates for surface V1."""

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from audit_surface_repetition import WORD
from compare_v12_fineweb_fresh import first_words
from document_proposition_plan import (
    FORMS, STEP_ACTIONS, edit_initial_fact, edit_update,
    sample_plan, validate,
)
from document_proposition_surface import (
    answer_from_visible_prefix, parse_visible_document, realize,
)
from document_world import NAMES, VALUES


def _target_site(plan, kind):
    index = 0
    for move in plan['moves']:
        if move['kind'] not in ('compare', 'step', 'revisit', 'question'):
            continue
        node = plan['nodes'][move['refs'][0]]
        if move['kind'] in ('revisit', 'question') and node['kind'] == kind:
            return index, node
        index += 1
    return None


def _comparison_site(plan):
    index = 0
    for move in plan['moves']:
        if move['kind'] == 'compare':
            return index, plan['nodes'][move['refs'][0]]
        if move['kind'] in ('step', 'revisit', 'question'):
            index += 1
    return None


def _edit_literal(plan, node_id, new_value):
    edited = copy.deepcopy(plan)
    assert edited['nodes'][node_id]['value'] != new_value
    edited['nodes'][node_id]['value'] = new_value
    for move in edited['moves']:
        if move['kind'] == 'question':
            move['answer'] = edited['nodes'][move['refs'][0]]['value']
    assert validate(edited)
    return edited


def _check_flip(plan, edited, site_index, surface_seed, canonical,
                form_counts, label, surface_version=1):
    for lexical in (False, True):
        before = (realize(plan, seed=surface_seed,
                          lexical_exponent=0., surface_version=surface_version)
                  if lexical else canonical)
        after = realize(edited, seed=surface_seed,
                        lexical_exponent=0. if lexical else None,
                        surface_version=surface_version)
        old_site, new_site = (before.sites[site_index],
                              after.sites[site_index])
        assert old_site.answer != new_site.answer
        assert answer_from_visible_prefix(
            after.tokens[:new_site.index], lexical) == new_site.answer
        parse_visible_document(after.tokens, lexical)
        form_counts[label] += 1


def semantic(seed=345, per_form=200, surface_version=1,
             planning_style='independent', fixed_moves=None):
    rng = random.Random(seed)
    counts = Counter()
    by_form = {}
    for form in FORMS:
        form_counts = Counter()
        for i in range(per_form):
            plan = sample_plan(rng, form=form, coupling_rate=1.,
                               question_rate=1.,
                               planning_style=planning_style,
                               n_moves=fixed_moves)
            assert validate(plan)
            surface_seed = rng.randrange(2**32)
            canonical = realize(plan, seed=surface_seed,
                                surface_version=surface_version)
            state = parse_visible_document(canonical.tokens)
            assert len(state.updates) == sum(
                m['kind'] == 'update' for m in plan['moves'])
            assert len(state.measures) == sum(
                m['kind'] == 'measure' for m in plan['moves'])
            assert len(state.steps) == sum(
                m['kind'] == 'step' for m in plan['moves'])
            for lexical in (False, True):
                doc = (realize(plan, seed=surface_seed,
                               lexical_exponent=0.,
                               surface_version=surface_version) if lexical
                       else canonical)
                parse_visible_document(doc.tokens, lexical)
                for site in doc.sites:
                    assert doc.tokens[site.index] == site.answer
                    assert answer_from_visible_prefix(
                        doc.tokens[:site.index], lexical) == site.answer
                    form_counts['original_sites'] += 1
                    form_counts['site_' + site.family] += 1
                form_counts['documents'] += 1
            if 'same' in canonical.tokens or 'different' in canonical.tokens:
                altered = list(canonical.tokens)
                idx = next(j for j, token in enumerate(altered)
                           if token in ('same', 'different'))
                altered[idx] = ('different' if altered[idx] == 'same'
                                else 'same')
                try:
                    parse_visible_document(altered)
                except AssertionError:
                    form_counts['false_comparisons_rejected'] += 1
                else:
                    raise AssertionError('False comparison accepted')
            numbered = [site for site in canonical.sites
                        if site.family == 'step_number']
            if len(numbered) >= 2 and numbered[1].answer == '2':
                first, second = numbered[:2]
                start = first.index
                while start and canonical.tokens[start-1] != '.':
                    start -= 1
                end = first.index
                while canonical.tokens[end] != '.':
                    end += 1
                edited_prefix = (canonical.tokens[:start] +
                                 canonical.tokens[end+1:second.index])
                try:
                    revised = answer_from_visible_prefix(edited_prefix)
                except (AssertionError, KeyError, IndexError):
                    revised = None  # A later clause may rely on step 1.
                if revised is not None:
                    assert revised == '1'
                    form_counts['supported_step_deletion_flips'] += 1
            target = _target_site(plan, 'update')
            if target:
                site_index, node = target
                future = [candidate['value'] for candidate in plan['nodes']
                          if candidate['kind'] == 'update' and
                          candidate['id'] > node['id'] and
                          (candidate['entity'], candidate['attr']) ==
                          (node['entity'], node['attr'])]
                choices = [value for value in VALUES[node['attr']]
                           if value not in (node['old'], node['value']) and
                           (not future or value != future[0])]
                if choices:
                    edited = edit_update(plan, node['id'], rng.choice(choices))
                    _check_flip(plan, edited, site_index, surface_seed,
                                canonical, form_counts,
                                'targeted_update_flips', surface_version)
            target = _target_site(plan, 'source_claim')
            if target:
                site_index, node = target
                edited = _edit_literal(plan, node['id'], rng.choice(
                    [name for name in NAMES if name != node['value']]))
                _check_flip(plan, edited, site_index, surface_seed,
                            canonical, form_counts,
                            'targeted_source_flips', surface_version)
            for kind, pool in (
                    ('measure', range(1, 31)),
                    ('record', None),
                    ('step', STEP_ACTIONS)):
                target = _target_site(plan, kind)
                if not target:
                    continue
                site_index, node = target
                if kind == 'record':
                    pool = {'status': ('active', 'closed', 'pending'),
                            'category': ('standard', 'special'),
                            'code': tuple(plan['aliases'].values())}[
                                node['key']]
                alternatives = [value for value in pool
                                if value != node['value']]
                if not alternatives:
                    continue
                edited = _edit_literal(plan, node['id'],
                                       rng.choice(alternatives))
                _check_flip(plan, edited, site_index, surface_seed,
                            canonical, form_counts,
                            'targeted_' + kind + '_flips', surface_version)
            comparison = _comparison_site(plan)
            if comparison:
                site_index, relation = comparison
                left, right = relation['deps']
                target_node = plan['nodes'][left]
                other_value = plan['nodes'][right]['value']
                candidates = ([other_value] if relation['value'] ==
                              'different' else
                              [v for v in VALUES[relation['attr']]
                               if v != other_value])
                for candidate in candidates:
                    if candidate == target_node['value']:
                        continue
                    try:
                        if target_node['kind'] == 'fact':
                            edited = edit_initial_fact(
                                plan, target_node['entity'],
                                target_node['attr'], candidate)
                        else:
                            edited = edit_update(plan, left, candidate)
                    except AssertionError:
                        continue
                    if edited['nodes'][relation['id']]['value'] != \
                            relation['value']:
                        _check_flip(plan, edited, site_index,
                                    surface_seed, canonical, form_counts,
                                    'targeted_comparison_flips',
                                    surface_version)
                        break
        by_form[form] = dict(form_counts)
        counts.update(form_counts)
    return dict(seed=seed, per_form=per_form,
                totals=dict(counts), by_form=by_form)


def form(seed, n=64, words=100, surface_version=1,
         planning_style='independent', fixed_moves=None):
    rng = random.Random(seed)
    texts = dict(canonical=[], typed=[])
    draws = 0
    while len(texts['typed']) < n:
        draws += 1
        plan = sample_plan(rng, coupling_rate=.7,
                           planning_style=planning_style,
                           n_moves=(fixed_moves if fixed_moves is not None
                                    else rng.randint(10, 16)))
        surface_seed = rng.randrange(2**32)
        docs = dict(canonical=realize(plan, seed=surface_seed,
                                      surface_version=surface_version),
                    typed=realize(plan, seed=surface_seed,
                                  lexical_exponent=0.,
                                  surface_version=surface_version))
        crops = {key: first_words(doc.text(), words)
                 for key, doc in docs.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        for key in texts:
            texts[key].append(crops[key])
    panels = {}
    for key, rows in texts.items():
        panel = structure(rows)
        documents = defaultdict(set)
        for doc_id, row in enumerate(rows):
            terms = [term.lower() for term in WORD.findall(row)]
            for index in range(len(terms)-2):
                documents[tuple(terms[index:index+3])].add(doc_id)
        panel['top_reused_trigrams_by_document_count'] = [
            dict(trigram=' '.join(trigram), documents=len(ids))
            for trigram, ids in sorted(
                documents.items(), key=lambda entry: (
                    -len(entry[1]), entry[0]))[:20]]
        panels[key] = panel
    return dict(seed=seed, draws=draws, panels=panels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--surface-version', type=int, choices=(1, 2), default=1)
    p.add_argument('--planning-style', choices=('independent', 'threaded'),
                   default='independent')
    p.add_argument('--fixed-moves', type=int)
    args = p.parse_args()
    result = dict(surface_version=args.surface_version,
                  planning_style=args.planning_style,
                  fixed_moves=args.fixed_moves,
                  semantic=semantic(surface_version=args.surface_version,
                                    planning_style=args.planning_style,
                                    fixed_moves=args.fixed_moves),
                  form=[form(94121, surface_version=args.surface_version,
                             planning_style=args.planning_style,
                             fixed_moves=args.fixed_moves),
                        form(94122, surface_version=args.surface_version,
                             planning_style=args.planning_style,
                             fixed_moves=args.fixed_moves)],
                  caveat='First semantic surface fixture, not a realistic '
                         'training distribution or a full-token oracle.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('semantic', result['semantic']['totals'])
    for panel in result['form']:
        for key, row in panel['panels'].items():
            print(panel['seed'], key, row['word_types'],
                  round(row['repeated_across_docs_trigram_occurrence_share'],
                        3), row['sentence_start_types'])


if __name__ == '__main__':
    main()
