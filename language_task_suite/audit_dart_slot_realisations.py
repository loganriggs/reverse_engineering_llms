"""Screen DART annotations as reusable, semantically slotted realizations.

Only aggregate statistics are written. Exact string coverage is necessary
for safe substitution, but does not prove that a sentence is faithful to its
triples or that replacing an entity preserves grammatical agreement.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path('/home/loganriggs/Coding/neuron_circuits/'
                      'artifacts/dart_v111/train.json')


def _entities(triples):
    entities = sorted({value.strip(' "') for triple in triples
                       for value in (triple[0], triple[2])
                       if value and not value.startswith('[')},
                      key=lambda x: (-len(x), x))
    if any(len(value) < 2 for value in entities):
        return None
    return entities


def _template(text, entities):
    """Return slot text only if all entity strings appear exactly once."""
    spans = []
    for slot, value in enumerate(entities):
        matches = list(re.finditer(re.escape(value), text, re.I))
        if len(matches) != 1:
            return None
        spans.append((*matches[0].span(), slot))
    spans.sort()
    if any(right > next_left for (_, right, _),
           (next_left, _, _) in zip(spans, spans[1:])):
        return None
    pieces, cursor = [], 0
    for left, right, slot in spans:
        pieces.extend((text[cursor:left], f'<{slot}>'))
        cursor = right
    pieces.append(text[cursor:])
    return re.sub(r'\s+', ' ', ''.join(pieces)).strip().lower()


def audit(source):
    rows = json.loads(source.read_text())
    by_source = defaultdict(Counter)
    signatures = defaultdict(lambda: defaultdict(set))
    for row in rows:
        triples = row['tripleset']
        entities = _entities(triples)
        signature = tuple(sorted(str(triple[1]) for triple in triples))
        for annotation in row['annotations']:
            name = annotation['source']
            counts = by_source[name]
            counts['annotations'] += 1
            counts[f'triples/{len(triples)}'] += 1
            text = annotation['text']
            counts['word_sum'] += len(re.findall(r"\b\w+\b", text))
            if not entities:
                counts['invalid_entities'] += 1
                continue
            found = sum(bool(re.search(re.escape(value), text, re.I))
                        for value in entities)
            if found == len(entities):
                counts['all_entities_mentioned'] += 1
            if len(triples) < 2:
                continue
            template = _template(text, entities)
            if template is None:
                continue
            counts['exact_once_all_entities_multi_triple'] += 1
            signatures[name][signature].add(template)
    result = {}
    for name, counts in by_source.items():
        family = signatures[name]
        usable = [len(templates) for templates in family.values()]
        result[name] = dict(counts=dict(counts),
                            multi_triple_predicate_signatures=len(family),
                            signatures_with_10_templates=sum(
                                size >= 10 for size in usable),
                            templates_total=sum(usable),
                            top_signatures=[
                                dict(predicates=list(signature),
                                     distinct_templates=len(templates))
                                for signature, templates in sorted(
                                    family.items(),
                                    key=lambda pair: -len(pair[1]))[:12]])
    return dict(source=str(source), records=len(rows), by_source=result,
                caveat='Exact string coverage is not semantic faithfulness. '
                       'One-shot slot replacement can break agreement, '
                       'coreference and discourse. DART is sentence-level '
                       'and not the project-report DGP.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, value in result['by_source'].items():
        counts = value['counts']
        print(name, counts['annotations'],
              counts.get('exact_once_all_entities_multi_triple', 0),
              value['signatures_with_10_templates'], flush=True)


if __name__ == '__main__':
    main()
