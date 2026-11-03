"""Independent factual audit of V10 act sentences and answer edits."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from audit_document_world_v5_clause_realizations import _edited_world
from document_world import replay
from document_world_v10_fact_acts import (
    _is_act, answer_from_visible_prefix, draw_varied, extra_sentences,
    realize)


def _parse_act(tokens):
    """Extract claims from the 12 V10 act forms, independently of rendering."""
    s = list(tokens)
    if s[:2] == ['Meanwhile', ',']:
        obj = s[3]
        if s[4:6] == ['belonged', 'to']:
            attr, value = 'owner', s[6]
        elif len(s) == 7:
            attr, value = 'color', s[5]
        else:
            assert s[4] == 'was' and s[6] == 'the'
            attr, value = 'place', s[7]
        return dict(kind='state_fact', object=obj,
                    attribute=attr, value=value)
    if s[:3] == ['The', 'ledger', 'listed']:
        assert s[3:4] == ['the'] and s[5:6] == ['of'] and s[7:8] == ['as']
        return dict(kind='state_fact', object=s[6],
                    attribute=s[4], value=s[8])
    if s[:4] == ['As', 'of', 'that', 'report']:
        assert s[4] == ',' and s[6] == 'had' and s[8:10] == ['as', 'its']
        return dict(kind='state_fact', object=s[5],
                    attribute=s[10], value=s[7])
    if s[:1] == ['Before']:
        assert s[1].endswith("'s") and s[2:5] == ['report', ',', 'the']
        assert s[6] == 'had' and s[8:10] == ['as', 'its']
        assert s[11:15] == [';', 'afterward', 'it', 'had']
        return dict(kind='contrast', object=s[5],
                    attribute=s[10], old=s[7], value=s[15],
                    source=s[1][:-2])
    if s[:3] == ['That', 'change', 'replaced']:
        assert s[4] == 'with' and s[6:8] == ['for', 'the']
        assert s[9] == 'of'
        return dict(kind='contrast', object=s[10],
                    attribute=s[8], old=s[3], value=s[5])
    if s[:4] == ['In', 'the', 'revision', ',']:
        assert s[4] == 'the' and s[6] == 'of'
        assert s[8:10] == ['went', 'from'] and s[11] == 'to'
        return dict(kind='contrast', object=s[7],
                    attribute=s[5], old=s[10], value=s[12])
    if s[:3] == ['For', 'comparison', ',']:
        assert s[4] == 'had' and s[6] == 'while' and s[8] == 'had'
        assert s[10] == 'for'
        return dict(kind='relation', object=s[3], value=s[5],
                    other=s[7], other_value=s[9], attribute=s[11])
    if s[:4] == ['The', 'two', 'entries', 'listed']:
        assert s[5] == 'as' and s[7] == 'and' and s[9] == 'as'
        assert s[11] == 'for'
        return dict(kind='relation', object=s[4], value=s[6],
                    other=s[8], other_value=s[10], attribute=s[12])
    if s[:2] == ['Separately', ',']:
        assert s[2] == 'the' and s[4:6] == ['values', 'were']
        assert s[7] == ':' and s[9] == ';' and s[11] == ':'
        return dict(kind='relation', object=s[6], value=s[8],
                    other=s[10], other_value=s[12], attribute=s[3])
    if s[:2] == ['A', 'prior']:
        assert s[3:5] == ['report', 'on'] and s[6:8] == ['came', 'from']
        assert s[9:14] == [';', 'this', 'one', 'came', 'from']
        return dict(kind='source_history', object=s[5],
                    attribute=s[2], old_source=s[8], source=s[14])
    if s[:3] == ['The', 'report', 'from']:
        assert s[4] == 'followed' and s[5].endswith("'s")
        assert s[6] == 'earlier' and s[8:10] == ['report', 'on']
        return dict(kind='source_history', object=s[10],
                    attribute=s[7], old_source=s[5][:-2], source=s[3])
    assert s[:1] == ['After'] and s[1].endswith("'s")
    assert s[2:4] == ['account', ','] and s[5:8] == ['filed', 'the', 'new']
    assert s[9:11] == ['report', 'on']
    return dict(kind='source_history', object=s[11],
                attribute=s[8], old_source=s[1][:-2], source=s[4])


def check_claims(world, form_entropy):
    sentences = extra_sentences(world, form_entropy)
    current = replay(world['initial'], [])
    sources = {}
    expected_plan = {a['after']: a for a in world['fact_act_plan']}
    counts = Counter()
    for j, (event, source) in enumerate(zip(
            world['events'], world['sources'], strict=True)):
        before = replay(current, [])
        key = (event['object'], event['attribute'])
        old_source = sources.get(key)
        current[event['object']][event['attribute']] = event['value']
        sources[key] = source
        if j not in expected_plan:
            assert j not in sentences
            continue
        assert len(sentences[j]) == 1
        assert _is_act(sentences[j][0][:-1])
        claim = _parse_act(sentences[j][0])
        plan = expected_plan[j]
        assert claim['kind'] == plan['kind']
        assert claim['object'] == plan['object']
        assert claim['attribute'] == plan['attribute']
        obj, attr = plan['object'], plan['attribute']
        if claim['kind'] == 'state_fact':
            assert claim['value'] == current[obj][attr]
        elif claim['kind'] == 'contrast':
            assert (obj, attr) == key
            assert claim['old'] == before[obj][attr]
            assert claim['value'] == current[obj][attr]
            if 'source' in claim:
                assert claim['source'] == source
        elif claim['kind'] == 'relation':
            assert claim['other'] == plan['other'] != obj
            assert claim['value'] == current[obj][attr]
            assert claim['other_value'] == current[plan['other']][attr]
        else:
            assert claim['kind'] == 'source_history'
            assert old_source is not None
            assert claim['old_source'] == old_source
            assert claim['source'] == source
        counts[claim['kind']] += 1
    return counts


def check_sites(doc, lexical):
    for site in doc.sites:
        assert doc.tokens[site.index] == site.answer
        assert site.answer in site.candidates
        assert answer_from_visible_prefix(
            doc.tokens[:site.index], lexical_enabled=lexical) == site.answer


def audit(n, seed, lexical_exponent):
    rng = random.Random(seed)
    families, edits, acts = Counter(), Counter(), Counter()
    for _ in range(n):
        doc = draw_varied(rng, act_rate=.8, form_entropy=1.,
                          lexical_exponent=lexical_exponent,
                          syntax_entropy=1., extra_source_recap_rate=.5,
                          source_delay_rate=.5)
        lexical = lexical_exponent is not None
        check_sites(doc, lexical)
        acts.update(check_claims(doc.latent, 1.))
        families.update(site.family for site in doc.sites)
        no_acts = realize(doc.latent, act_enabled=False,
                          lexical_enabled=lexical, syntax_entropy=1.)
        assert [(s.family, s.answer) for s in no_acts.sites] == [
            (s.family, s.answer) for s in doc.sites]
        check_sites(no_acts, lexical)
        for j, plan in enumerate(doc.latent['recap_plan']):
            edited = realize(_edited_world(doc.latent, plan),
                             lexical_enabled=lexical, syntax_entropy=1.)
            check_sites(edited, lexical)
            assert edited.sites[j].family == doc.sites[j].family
            assert edited.sites[j].answer != doc.sites[j].answer
            edits[plan[1]] += 1
    return dict(n=n, seed=seed, lexical_exponent=lexical_exponent,
                sites=dict(families), supported_edits=dict(edits),
                fact_acts=dict(acts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--seed', type=int, default=86121)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    panels = [audit(a.n, a.seed+i, lexical)
              for i, lexical in enumerate((None, 0.))]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(panels=panels), indent=2) + '\n')
    for panel in panels:
        print(panel)


if __name__ == '__main__':
    main()
