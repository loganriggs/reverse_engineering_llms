"""Audit manually delexicalized WebNLG as a reusable semantic-form bank.

Source XML remains local. The exported JSON contains aggregate counts only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

from audit_document_structure_vs_fineweb import structure


DEFAULT_ROOT = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/'
                    'enriched_webnlg/data/v1.6/en/train')
TAG = re.compile(r'\b(?:AGENT|PATIENT|BRIDGE)-\d+\b')
NUMBER = re.compile(r'(?<!\w)\d[\d,./-]*(?!\w)')


def _triples(entry):
    return [tuple(part.strip() for part in (node.text or '').split(' | '))
            for node in entry.findall('./modifiedtripleset/mtriple')]


def _entitymap(entry):
    result = {}
    for node in entry.findall('./entitymap/entity'):
        text = node.text or ''
        if ' | ' in text:
            tag, entity = text.split(' | ', 1)
            result[entity.strip()] = tag.strip()
    return result


def _graph_key(entry, triples, mapping):
    edges = tuple(sorted((mapping.get(s, '?'), p, mapping.get(o, '?'))
                         for s, p, o in triples))
    return entry.get('category', ''), edges


def audit(root, seed=75961, panel_n=512):
    rng = random.Random(seed)
    files = sorted(root.glob('*triples/*.xml'))
    counts = Counter()
    by_category = defaultdict(Counter)
    graph_entries = defaultdict(set)
    graph_forms = defaultdict(set)
    category_texts = defaultdict(list)
    root_entries = defaultdict(list)
    file_hashes = {}
    for path in files:
        file_hashes[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()).hexdigest()
        tree = ET.parse(path)
        for entry in tree.findall('.//entry'):
            counts['entries'] += 1
            category = entry.get('category', '')
            by_category[category]['entries'] += 1
            triples = _triples(entry)
            mapping = _entitymap(entry)
            if len(triples) != int(entry.get('size', '-1')):
                counts['size_mismatch'] += 1
            if any(s not in mapping or o not in mapping
                   for s, _, o in triples):
                counts['unmapped_graph_entity'] += 1
            key = _graph_key(entry, triples, mapping)
            entry_id = (str(path.relative_to(root)), entry.get('eid', ''))
            graph_entries[key].add(entry_id)
            root_entity = next((entity for entity, tag in mapping.items()
                                if tag == 'AGENT-1'), None)
            if root_entity:
                root_entries[(category, root_entity)].append(
                    frozenset(triples))
            for lex in entry.findall('./lex'):
                counts['lexicalizations'] += 1
                by_category[category]['lexicalizations'] += 1
                comment = lex.get('comment', '')
                counts['comment:'+comment] += 1
                template = (lex.findtext('template') or '').strip()
                text = (lex.findtext('text') or '').strip()
                if not template or not text:
                    counts['missing_text_or_template'] += 1
                    continue
                tags = set(TAG.findall(template))
                mapped = set(mapping.values())
                if tags <= mapped:
                    counts['all_template_tags_mapped'] += 1
                if tags == mapped:
                    counts['all_graph_entities_in_template'] += 1
                if NUMBER.search(TAG.sub('', template)):
                    counts['unbound_template_number'] += 1
                if comment == 'good':
                    counts['good_with_template'] += 1
                    graph_forms[key].add(template)
                    category_texts[category].append(text)
    reusable = {key for key, entries in graph_entries.items()
                if len(entries) >= 2}
    repeated_keys = sum(len(graph_entries[key]) for key in reusable)
    multi_roots = {key: list(set(graphs)) for key, graphs in
                   root_entries.items() if len(set(graphs)) >= 2}
    disjoint_roots = {key for key, graphs in multi_roots.items()
                      if any(a.isdisjoint(b) for i, a in enumerate(graphs)
                             for b in graphs[i+1:])}
    panels = {}
    for category, texts in sorted(category_texts.items()):
        sample = rng.sample(texts, min(panel_n, len(texts)))
        panels[category] = {
            **by_category[category],
            'good_texts': len(texts),
            'panel_n': len(sample),
            'form': structure(sample) if sample else None}
    counts.update({
        'xml_files': len(files),
        'distinct_role_graph_keys': len(graph_entries),
        'role_graph_keys_with_two_entries': len(reusable),
        'entries_in_reusable_role_graph_keys': repeated_keys,
        'reusable_keys_with_two_good_templates': sum(
            len(graph_forms[key]) >= 2 for key in reusable),
        'distinct_root_entities': len(root_entries),
        'roots_with_two_distinct_graphs': len(multi_roots),
        'roots_with_two_disjoint_graphs': len(disjoint_roots)})
    return {
        'root': str(root), 'seed': seed, 'panel_n': panel_n,
        'counts': dict(counts),
        'categories': panels,
        'source_file_sha256': file_hashes,
        'caveat': 'Role-graph equality permits a candidate template swap, '
                  'but does not prove grammar, factuality, or discourse '
                  'appropriateness after substitution. The corpus is '
                  'sentence/short-text data, not a document DGP.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--seed', type=int, default=75961)
    parser.add_argument('--panel-n', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.seed, args.panel_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'])


if __name__ == '__main__':
    main()
