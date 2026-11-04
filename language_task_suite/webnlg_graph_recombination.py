"""Prototype novel triple graphs with corpus-attested WebNLG surface forms.

This is deliberately a candidate generator. Structural checks establish
slot/graph consistency, not the English truth conditions of every template.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
import re
from pathlib import Path
import xml.etree.ElementTree as ET

from audit_enriched_webnlg_reuse import (
    DEFAULT_ROOT, NUMBER, TAG, _entitymap, _graph_key, _triples,
)


@dataclass(frozen=True)
class Form:
    category: str
    graph_key: tuple
    triples: tuple[tuple[str, str, str], ...]
    roles: tuple[tuple[str, str], ...]
    template: str
    original_text: str


def load_forms(root=DEFAULT_ROOT):
    groups = defaultdict(list)
    for path in sorted(Path(root).glob('*triples/*.xml')):
        for entry in ET.parse(path).findall('.//entry'):
            triples = _triples(entry)
            mapping = _entitymap(entry)
            if not triples or any(s not in mapping or o not in mapping
                                  for s, _, o in triples):
                continue
            key = _graph_key(entry, triples, mapping)
            roles = tuple(sorted((tag, entity) for entity, tag in
                                 mapping.items()))
            for lex in entry.findall('./lex'):
                if lex.get('comment') != 'good':
                    continue
                template = (lex.findtext('template') or '').strip()
                original = (lex.findtext('text') or '').strip()
                if not template or not original:
                    continue
                if (set(TAG.findall(template)) != {tag for tag, _ in roles}
                        or NUMBER.search(TAG.sub('', template))):
                    continue
                groups[key].append(Form(entry.get('category', ''), key,
                                        tuple(triples), roles, template,
                                        original))
    return {key: forms for key, forms in groups.items()
            if len({form.roles for form in forms}) >= 2}


def _lexical_name(entity):
    entity = entity.split('@', 1)[0].strip('"')
    entity = re.sub(r'\s*\([^)]*\)$', '', entity)
    return entity.replace('_', ' ').replace(' ,', ',').strip()


def _clean(text):
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'([,;:])(?=[^\s])', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def render(template, role_values):
    used = set(TAG.findall(template))
    if used != set(role_values):
        raise ValueError('Template roles do not equal graph roles')
    text = TAG.sub(lambda match: role_values[match.group()], template)
    if TAG.search(text):
        raise AssertionError('Unfilled role tag')
    return _clean(text)


def sample(groups, rng: random.Random, *, cross_role_rate=.5):
    if not 0 <= cross_role_rate <= 1:
        raise ValueError('cross_role_rate outside [0,1]')
    key = rng.choice(tuple(groups))
    forms = groups[key]
    source = rng.choice(forms)
    targets = [form for form in forms if form.roles != source.roles]
    target = rng.choice(targets)
    role_values = {tag: _lexical_name(entity) for tag, entity in
                   target.roles}
    crossed_tag = None
    if rng.random() < cross_role_rate:
        donor = rng.choice([form for form in forms
                            if form.roles != target.roles])
        donor_values = {tag: _lexical_name(entity) for tag, entity in
                        donor.roles}
        eligible = [tag for tag in role_values
                    if tag.startswith(('PATIENT-', 'BRIDGE-')) and
                    donor_values[tag] != role_values[tag]]
        if eligible:
            crossed_tag = rng.choice(eligible)
            role_values[crossed_tag] = donor_values[crossed_tag]
    # Graph roles are the same for every form in the key. This constructs
    # the actual sampled world, including the crossed value if any.
    triples = tuple((role_values[s], predicate, role_values[o])
                    for s, predicate, o in key[1])
    text = render(source.template, role_values)
    assert len(triples) == len(source.triples)
    return {'category': source.category, 'graph_key': key,
            'triples': triples, 'roles': role_values,
            'text': text, 'source_text': source.original_text,
            'template': source.template,
            'crossed_tag': crossed_tag}
