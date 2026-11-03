"""Build the local, self-refreshing DGP viewer data file.

The generated JavaScript embeds short excerpts from natural source documents
and is deliberately Git-ignored. The committed HTML shell has no corpus text.
Run once to refresh, or use --watch while actively editing DGP code/results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys
import time

from tokenizers import Tokenizer

from audit_corpus_joint_style_delimiter import aligned
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_joint_style_delimiter_dgp import plans as web_plans
from corpus_joint_style_delimiter_dgp import validate_factorial
from document_mixture_v7_cross import KINDS, sample_document, site_oracle
from sports_alias_bracket_dgp import realize as sports_realize
from sports_alias_bracket_dgp import sample_plan as sports_sample_plan


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/'
                     'dgp_viewer/fineweb_sources.json')
ARM_ORDER = (('us', 'round'), ('uk', 'round'),
             ('us', 'square'), ('uk', 'square'))


def _json(path):
    return json.loads((HERE/path).read_text())


def _site(site):
    return {'kind': site.kind, 'start': site.start, 'end': site.end,
            'answer': site.answer, 'source_start': site.source_start,
            'source_end': site.source_end}


def _render_tokens(tokens):
    output, spans = '', []
    tight_before = {'.', ',', ';', ':', '!', '?', ')', ']', '%', "'s"}
    tight_after = {'(', '[', '$'}
    for word in tokens:
        if word == '\n':
            output = output.rstrip()+'\n'
            spans.append((len(output)-1, len(output)))
            continue
        if output and not output.endswith(('\n', ' ')) and \
                word not in tight_before and output[-1] not in tight_after:
            output += ' '
        start = len(output)
        output += word
        spans.append((start, len(output)))
    return output, spans


def _synthetic_examples():
    names = {
        'narrative': 'Narrative: lexical, action, and attribute rules',
        'event_report': 'Event report: aliases and visible world state',
        'record_listing': 'Record listing: scoped lookup',
        'score_report': 'Score report: state update',
        'cross_report': 'Cross report: source → reviewer → record',
    }
    examples = []
    for index, kind in enumerate(KINDS):
        doc = sample_document(random.Random(98630+index), kind=kind)
        text, spans = _render_tokens(doc.tokens)
        sites = []
        for site in doc.sites:
            oracle = site_oracle(doc, site)
            assert oracle.get(site.target, 0) > 0
            start, end = spans[site.index]
            sites.append({'kind': site.family, 'start': start, 'end': end,
                          'answer': site.target, 'source_start': None,
                          'source_end': None})
        examples.append({
            'id': 'symbolic_'+kind, 'group': 'symbolic',
            'title': names[kind], 'kind': kind,
            'source': 'Sampled V7 task-rich document grammar',
            'status': ('The generated target has an independently checked '
                       'visible-prefix oracle. V7 surface form remains far '
                       'from natural web text; harder transfer rules were '
                       'not behaviorally learned.'),
            'report': 'DOCUMENT_MIXTURE_V7_CROSS_AND_TRANSFER.md',
            'arms': [{'id': 'sample', 'label': 'Sampled document',
                      'text': text, 'sites': sites}],
        })
    return examples


def _sports_example():
    plan = sports_sample_plan(random.Random(98601), 'heldout')
    arms = []
    for alias in ('a', 'b'):
        for bracket in ('round', 'square'):
            doc = sports_realize(plan, alias, bracket)
            arms.append({'id': alias+'_'+bracket,
                         'label': ('Alias '+alias.upper()+' · '+bracket),
                         'text': doc.text,
                         'sites': [_site(site) for site in doc.sites]})
    fresh = _json(Path('results/sports_alias_bracket_fresh_seed98301_n100.json'))
    metrics = []
    for row in fresh['results']:
        m = row['metrics']
        metrics.append({'alias_paired_exact': m['alias']['paired_top1'],
                        'alias_paired_two_choice': m['alias']['candidate_paired'],
                        'bracket_paired_exact': m['bracket']['paired_top1']})
    return {'id': 'sports_alias_bracket', 'group': 'natural',
            'title': 'Sports report: name reuse and bracket closure',
            'kind': 'sports_alias_bracket',
            'source': 'Human RotoWire report with two controlled edits',
            'source_hash': hashlib.sha256(
                repr(plan.alias.game_key).encode()).hexdigest()[:12],
            'status': ('Two depth-2 bilinear models learned bracket closure '
                       'strongly. Alias copying responds to edits but is '
                       'incomplete; do not infer an exact copy circuit.'),
            'report': 'SPORTS_ALIAS_BRACKET_JOINT_PILOT.md',
            'metrics': metrics, 'arms': arms}


def _fineweb_cache(cache_path, refresh):
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    from datasets import load_dataset
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    examples = []
    stream = iter(source)
    try:
        for offset, row in enumerate(islice(stream, 83500, 84500), 83500):
            text = row['text'][:8000]
            for plan in sorted(web_plans(text),
                               key=lambda item: item.joint_span_characters):
                arms = validate_factorial(plan)
                if aligned(arms, tokenizer, 320) is None:
                    continue
                examples.append({'offset': offset, 'text': text,
                                 'style_occurrence_index':
                                 plan.style_occurrence_index,
                                 'delimiter_pair_index':
                                 plan.delimiter_pair_index})
                break
            if len(examples) >= 3:
                break
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    if len(examples) < 2:
        raise RuntimeError('Too few aligned FineWeb joint examples')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(examples, ensure_ascii=False))
    return examples


def _fineweb_examples(cache_path, refresh):
    from corpus_joint_style_delimiter_dgp import Plan
    rows = _fineweb_cache(cache_path, refresh)
    examples = []
    for index, row in enumerate(rows):
        plan = Plan(row['text'], row['style_occurrence_index'],
                    row['delimiter_pair_index'], 0)
        docs = validate_factorial(plan)
        arms = []
        for style, bracket in ARM_ORDER:
            doc = docs[style, bracket]
            arms.append({'id': style+'_'+bracket,
                         'label': style.upper()+' spelling · '+bracket,
                         'text': doc.text,
                         'sites': [_site(site) for site in doc.sites]})
        examples.append({
            'id': 'fineweb_style_bracket_'+str(index), 'group': 'natural',
            'title': 'Web document: spelling convention and delimiter '+
                     str(index+1), 'kind': 'fineweb_style_bracket',
            'source': 'FineWeb prose with two controlled edits',
            'source_hash': hashlib.sha256(row['text'].encode()).hexdigest()[:12],
            'status': ('Four-arm generator and answer checks pass. Only '
                       '18/1,000 source documents aligned in the audit; '
                       'no joint model has been trained. Earlier '
                       'spelling-only models failed held-out rule checks.'),
            'report': 'CORPUS_JOINT_RULE_FEASIBILITY.md',
            'arms': arms})
    return examples


def _catalog():
    return [
        {'name': 'Adjacent bigram', 'rule': 'The immediately preceding token '
         'changes the next-token distribution.', 'source': 'Sampled V3 grammar',
         'example': 'noun → next word; learned in the crossed-skip control'},
        {'name': 'Crossed skip rule', 'rule': 'An earlier place and adjacent '
         'adjective jointly select one of four outputs. Neither cue alone '
         'identifies the answer.', 'source': 'Sampled V3 grammar',
         'example': 'lake … new → and; beach … new → or'},
        {'name': 'Induction / copying', 'rule': 'A later cue retrieves a token '
         'or identifier from an earlier occurrence.',
         'source': 'Sampled V3 grammar / natural name reuse',
         'example': 'full player name → later surname'},
        {'name': 'Bracket closure', 'rule': 'The visible unmatched opener '
         'determines the closer after intervening content.',
         'source': 'Sampled V3 / natural reports / FineWeb',
         'example': '( … ) or [ … ]'},
        {'name': 'Spelling convention', 'rule': 'An earlier spelling variant '
         'cues US or UK style for a different later lexical family.',
         'source': 'FineWeb controlled transformation',
         'example': 'color … center; colour … centre'},
        {'name': 'Source attribution', 'rule': 'The latest visible claim '
         'about an object and attribute identifies its source.',
         'source': 'Sampled V7 event reports',
         'example': 'later report replaces an earlier source'},
        {'name': 'Scoped record lookup', 'rule': 'A ship and room jointly '
         'identify a category; a ship name alone is insufficient.',
         'source': 'Sampled V7 records',
         'example': 'ship + room → category'},
        {'name': 'Score update', 'rule': 'A visible score and event increment '
         'determine the later score.', 'source': 'Sampled V7 scores',
         'example': 'previous score + increment → current score'},
        {'name': 'Cross-section composition', 'rule': 'Find the latest '
         'source, map that source to a reviewer ship, then look up the '
         'ship’s category for the requested room.',
         'source': 'Sampled V7 cross report',
         'example': 'source → ship → (ship, room) → category'},
    ]


def build(cache_path, refresh):
    return {'generated_at': datetime.now(timezone.utc).isoformat(
                timespec='seconds'),
            'focus': ('Realistic, variable sampled language with known '
                      'rules; behaviorally verify each learned rule before '
                      'weight-based interpretation.'),
            'groups': [
                {'id': 'symbolic', 'title': 'Fully sampled rules',
                 'summary': 'Exact task answers and causal edits; form gap '
                            'relative to real web prose.'},
                {'id': 'natural', 'title': 'Natural-text carriers',
                 'summary': 'Real prose with controlled rule sites; no full '
                            'next-token oracle for background text.'},
                {'id': 'candidate', 'title': 'Candidate, not trained',
                 'summary': 'Uppercase recurrence plus a delimiter yields '
                            '58/1,000 span-aligned FineWeb opportunities, '
                            'but replacement semantics need validation.'}],
            'catalog': _catalog(),
            'examples': (_synthetic_examples()+[_sports_example()]+
                         _fineweb_examples(cache_path, refresh)),
            'candidate': _json(Path('results/'
                                    'corpus_uppercase_repeat_joint_skip84500_n1000.json'))['counts'],
            'data_policy': ('Natural source passages live only in the local '
                            'Git-ignored data file. The committed viewer '
                            'contains no source text.')}


def write_data(path, cache_path, refresh):
    data = build(cache_path, refresh)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text('window.DGP_VIEWER_DATA = '+
                         json.dumps(data, ensure_ascii=False).replace(
                             '</', '<\\/')+';\n')
    os.replace(temporary, path)
    print('Updated', path, 'with', len(data['examples']),
          'sampled examples at', data['generated_at'], flush=True)


def _fingerprint():
    paths = [HERE/name for name in (
        'build_dgp_viewer.py', 'document_mixture_v7_cross.py',
        'sports_alias_bracket_dgp.py',
        'corpus_joint_style_delimiter_dgp.py',
        'results/sports_alias_bracket_fresh_seed98301_n100.json',
        'results/corpus_joint_style_delimiter_audit_skip83500_n1000.json',
        'results/corpus_uppercase_repeat_joint_skip84500_n1000.json')]
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                 for path in paths)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path,
                   default=HERE/'dgp_viewer_data.js')
    p.add_argument('--cache-path', type=Path, default=DEFAULT_CACHE)
    p.add_argument('--refresh-corpus', action='store_true')
    p.add_argument('--watch', action='store_true',
                   help='Rebuild when the DGP code or result files change')
    args = p.parse_args()
    write_data(args.output, args.cache_path, args.refresh_corpus)
    if args.watch:
        fingerprint = _fingerprint()
        while True:
            time.sleep(15)
            current = _fingerprint()
            if current != fingerprint:
                write_data(args.output, args.cache_path, False)
                fingerprint = current
    sys.stdout.flush()
    os._exit(0)  # streaming datasets may crash during interpreter shutdown


if __name__ == '__main__':
    main()
