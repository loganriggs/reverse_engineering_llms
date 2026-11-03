"""Build the local, self-refreshing DGP viewer data file.

The generated JavaScript embeds short excerpts from natural source documents
and is deliberately Git-ignored. The committed HTML shell has no corpus text.
Run once to refresh, or use --watch while actively editing DGP code/results.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from document_mixture_v7_cross import (KINDS, TASK_RICH_WEIGHTS,
                                       sample_document, site_oracle)
from document_world_v5_clause_realizations import (
    answer_from_visible_prefix, draw_varied, realize,
)
from sports_alias_bracket_dgp import realize as sports_realize
from sports_alias_bracket_dgp import sample_plan as sports_sample_plan
from audit_surface_repetition import WORD


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


def _renderer_variant_example():
    world = draw_varied(random.Random(98640), source_delay_rate=1.).latent
    arms = []
    expected = None
    for rate, label in ((0., 'Current event frames'),
                        (1., 'Alternate intro / recap frames')):
        doc = realize(world, source_delay_rate=1., variant_rate=rate)
        observed = [(site.family, site.answer) for site in doc.sites]
        if expected is None:
            expected = observed
        assert observed == expected
        rendered, spans = _render_tokens(doc.tokens)
        sites = []
        for site in doc.sites:
            assert answer_from_visible_prefix(doc.tokens[:site.index]) == site.answer
            start, end = spans[site.index]
            sites.append({'kind': site.family, 'start': start, 'end': end,
                          'answer': site.answer, 'source_start': None,
                          'source_end': None})
        arms.append({'id': 'rate_'+str(rate), 'label': label,
                     'text': rendered, 'sites': sites})
    assert arms[0]['text'] != arms[1]['text']
    return {'id': 'v7_renderer_variant', 'group': 'symbolic',
            'title': 'Same world, two event-report renderers',
            'kind': 'event_renderer_variant',
            'source': 'Paired V7 world; optional within-kind form variation',
            'status': ('Both arms preserve all visible-prefix answers. '
                       'The new fixed frames reduce “the” but increase '
                       'top-ten concentration at high use; this is a '
                       'negative lexical-realism result, not a trained '
                       'model result.'),
            'report': 'V7_RENDERER_VARIATION_PILOT.md', 'arms': arms}


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
                       'strongly on held-out RotoWire reports, but bracket '
                       'accuracy drops on position-balanced SBNation. '
                       'Alias copying responds to edits but is incomplete; '
                       'do not infer a portable exact-copy circuit.'),
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


def _zipf_comparison():
    """Equal-word-count panels, including the actual V7 document sampler."""
    frozen = _json(Path('results/v14_zipf_seed99851_n480/'
                        'v14_vs_fineweb_word_rank.json'))
    sports = _json(Path('results/natural_sports_zipf_n480.json'))
    rng = random.Random(99852)
    rows = []
    attempts = 0
    while len(rows) < 480 and attempts < 5000:
        attempts += 1
        kind = rng.choice(('narrative', 'event_report', 'cross_report'))
        doc = sample_document(rng, kind=kind)
        passage, _ = _render_tokens(doc.tokens)
        words = [word.lower() for word in WORD.findall(passage)]
        if len(words) >= 100:
            rows.append(words[:100])
    if len(rows) != 480:
        raise RuntimeError('Could not sample 480 long V7 windows')
    counts = Counter(word for row in rows for word in row)
    rank = sorted(counts.values(), reverse=True)
    total = sum(rank)
    # Match audit_v14_zipf.py's descriptive least-squares slope.
    import numpy as np
    slope = float(np.polyfit(np.log(np.arange(10, 101)),
                             np.log(np.asarray(rank[9:100])/total), 1)[0])
    v7 = {'n_windows': 480, 'words_per_window': 100,
          'total_words': total, 'observed_vocabulary': len(counts),
          'top1_mass': rank[0]/total, 'top10_mass': sum(rank[:10])/total,
          'tail_slope_rank10_to100': slope,
          'hapax_types': sum(value == 1 for value in counts.values()),
          'top25': counts.most_common(25)}
    # Training draws whole documents with TASK_RICH_WEIGHTS. Count a continuous
    # stream of those documents so short kinds are present in their actual
    # proportions and longer kinds contribute more word tokens, as in CE.
    stream_rng = random.Random(99853)
    stream_words = []
    kind_counts = Counter()
    while len(stream_words) < 48000:
        doc = sample_document(stream_rng, weights=TASK_RICH_WEIGHTS)
        kind_counts[doc.kind] += 1
        stream_words.extend(w.lower() for w in WORD.findall(' '.join(doc.tokens)))
    stream_words = stream_words[:48000]
    mix_counts = Counter(stream_words)
    mix_rank = sorted(mix_counts.values(), reverse=True)
    mix_slope = float(np.polyfit(np.log(np.arange(10, 101)),
                                 np.log(np.asarray(mix_rank[9:100])/48000),
                                 1)[0])
    mixture = {'n_windows': 480, 'words_per_window': 100,
               'total_words': 48000,
               'observed_vocabulary': len(mix_counts),
               'top1_mass': mix_rank[0]/48000,
               'top10_mass': sum(mix_rank[:10])/48000,
               'tail_slope_rank10_to100': mix_slope,
               'hapax_types': sum(v == 1 for v in mix_counts.values()),
               'top25': mix_counts.most_common(25)}
    return {'method': ('48,000 lowercase alphabetic words per arm, displayed '
                       'as 480 blocks of 100. The V7 training-mixture arm '
                       'uses 48,000 consecutive words from whole documents '
                       'sampled at the actual training kind weights; it '
                       'includes short kinds. The V7 long-document arm '
                       'conditions on eligible narrative/event/cross docs. '
                       'FineWeb and V14 use the '
                       'previously frozen reference audit. Human sports '
                       'reports use unique training games. Panels are '
                       'equal-sized, independent, and from different genres.'),
            'caveat': ('A near -1 middle-rank slope alone does not mean '
                       'Zipfian natural language: head mass and vocabulary '
                       'tail matter, and lexical frequency does not test '
                       'syntax or meaning.'),
            'series': [
                {'name': 'FineWeb reference', 'summary': frozen['summary']['FineWeb'],
                 'ranks': frozen['ranks']['FineWeb']},
                {'name': 'V7 training mixture', 'summary': mixture,
                 'ranks': mix_rank},
                {'name': 'V14 sampled grammar', 'summary': frozen['summary']['V14'],
                 'ranks': frozen['ranks']['V14']},
                {'name': 'V7 long-document sample', 'summary': v7,
                 'ranks': rank},
                *[{'name': 'Human '+name.upper(),
                   'summary': sports['panels'][name]['summary'],
                   'ranks': sports['panels'][name]['ranks']}
                  for name in ('rotowire', 'sbnation')]],
            'v7_training_weights': dict(zip(KINDS, TASK_RICH_WEIGHTS)),
            'v7_documents_sampled': dict(kind_counts),
            'reference': 'V14_SAMPLED_EXAMPLES_AND_ZIPF.md; '
                         'DOCUMENT_MIXTURE_V7_CROSS_AND_TRANSFER.md; '
                         'NATURAL_SPORTS_ZIPF_REFERENCE.md'}


def _action_path_summary():
    rows = []
    panels = ((34801, 35051, 3, 'v7_action_edge_patch_seed35051.json'),
              (34801, 35052, 3, 'v7_action_edge_patch_seed35052.json'),
              (34802, 35051, 1, 'v7_action_edge_patch_model34802.json'))
    for model_seed, panel_seed, head, filename in panels:
        result = _json(Path('results')/filename)['result']
        selected = {(row['layer'], row['head'], row['patch']): row
                    for row in result['rows']}
        def removed(mode):
            return selected[2, head, mode]['mean_response_removed']
        rows.append({'model_seed': model_seed, 'panel_seed': panel_seed,
                     'head': head, 'n': result['n'],
                     'native_response': result['native_response'],
                     'source_score': removed('score_edge'),
                     'source_value': removed('value_source'),
                     'score_and_value': removed('score_and_value'),
                     'preceding_score': removed('preceding_score_edge')})
    return {'rows': rows,
            'status': ('A local action edit is carried mainly by one '
                       'layer-2 attention head and its changed-token edge. '
                       'The dominant head changes across two model seeds. '
                       'This is an activation patch, not weight-only '
                       'circuit recovery.'),
            'report': 'V7_ACTION_ATTENTION_EDGE_PATCH.md'}


def _pair_reduction_summary():
    panels = {34801: ('v7_action_rotary_pair_seed35151.json',
                      'v7_action_rotary_pair_seed35251.json'),
              34802: ('v7_action_rotary_pair_model34802.json',
                      'v7_action_rotary_pair_model34802_rep.json')}
    rows = []
    for model_seed, filenames in panels.items():
        results = [_json(Path('results')/name) for name in filenames]
        for rank in (4, 6):
            for method in ('weight_only', 'unlabeled_other_kinds',
                           'supervised_action_development', 'random_control'):
                measures = [next(row for row in result['rows']
                                 if row['method'] == method and
                                 row['pairs_per_branch'] == rank)
                            for result in results]
                rows.append({'model_seed': model_seed, 'rank': rank,
                             'method': method,
                             'token_nll': sum(m['token_nll'] for m in measures)/2,
                             'action_accuracy': sum(m['action_accuracy']
                                                    for m in measures)/2,
                             'edit_response': sum(m['edit_response']
                                                  for m in measures)/2})
    return {'rows': rows,
            'status': ('Means over two fresh panels per independently trained '
                       'model. Weight-only ranking beats random at both '
                       'budgets, but does not recover a nearly complete '
                       'small circuit in the second model.'),
            'report': 'V7_ACTION_ROTARY_PAIR_REDUCTION.md'}


def build(cache_path, refresh):
    transfer = _json(Path('results/sports_alias_bracket_position_control_seed98351.json'))
    fresh600 = _json(Path('results/sports_alias_bracket_multistyle_fresh_seed98451.json'))
    fresh1200 = _json(Path('results/sports_alias_bracket_multistyle_1200_fresh_seed98451.json'))
    training_rows = []
    for index, seed in enumerate((98401, 98402)):
        mixed600 = _json(Path(f'results/sports_alias_bracket_multistyle_mixed_seed{seed}_600.json'))
        control = _json(Path(f'results/sports_alias_bracket_multistyle_rwcontrol_seed{seed}_600.json'))
        mixed1200 = _json(Path(f'results/sports_alias_bracket_multistyle_mixed_seed{seed}_1200.json'))
        for label, source, fresh in (
                ('Original', mixed600['trajectory'][0], fresh600['results']['original'][index]),
                ('RotoWire +600', control['trajectory'][-1], fresh600['results']['rotowire_control'][index]),
                ('Mixed +600', mixed600['trajectory'][-1], fresh600['results']['mixed'][index]),
                ('Mixed +1200', mixed1200['trajectory'][-1], fresh1200['results']['mixed'][index])):
            training_rows.append({
                'seed': seed, 'arm': label,
                'rw_bracket': source['matched']['rotowire']['bracket']['paired_top1'],
                'sb_bracket': source['matched']['sbnation']['bracket']['paired_top1'],
                'fresh_sb_bracket': fresh['metrics']['sbnation']['bracket']['paired_top1']})
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
            'zipf': _zipf_comparison(),
            'action_path': _action_path_summary(),
            'pair_reduction': _pair_reduction_summary(),
            'zipf_by_kind': _json(Path('results/v7_zipf_by_kind_n500.json')),
            'zipf_variant': [
                _json(Path('results/'+name)) for name in (
                    'v7_renderer_variant_rate025_n480.json',
                    'v7_renderer_variant_rate05_n480.json',
                    'v7_renderer_variant_n480.json')],
            'style_transfer': {
                'matched_n': transfer['panel']['matched_n_per_source'],
                'matched_per_bin': transfer['panel']['matched_per_bin'],
                'seeds': [{source: {kind: result['metrics'][source][
                    'matched'][kind] for kind in ('alias', 'bracket')}
                           for source in ('rotowire', 'sbnation')}
                          for result in transfer['results']],
                'caveat': transfer['caveat'],
                'report': 'SPORTS_ALIAS_BRACKET_STYLE_TRANSFER.md'},
            'style_training': {
                'rows': training_rows,
                'matched_n': 46, 'fresh_sbnation_n': 15,
                'status': ('Mixed-style exposure helps, but 1,200-step '
                           'models remain below the .8 paired-exact '
                           'cross-style behavior gate. No portable '
                           'bracket circuit claim.'),
                'report': 'SPORTS_ALIAS_BRACKET_MULTISTYLE_CONTINUATION.md'},
            'examples': (_synthetic_examples()+[_renderer_variant_example(),
                         _sports_example()]+
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
        'document_world_v5_clause_realizations.py',
        'sports_alias_bracket_dgp.py',
        'corpus_joint_style_delimiter_dgp.py',
        'results/sports_alias_bracket_fresh_seed98301_n100.json',
        'results/sports_alias_bracket_position_control_seed98351.json',
        'results/sports_alias_bracket_multistyle_fresh_seed98451.json',
        'results/sports_alias_bracket_multistyle_1200_fresh_seed98451.json',
        'results/sports_alias_bracket_multistyle_mixed_seed98401_600.json',
        'results/sports_alias_bracket_multistyle_mixed_seed98402_600.json',
        'results/sports_alias_bracket_multistyle_rwcontrol_seed98401_600.json',
        'results/sports_alias_bracket_multistyle_rwcontrol_seed98402_600.json',
        'results/sports_alias_bracket_multistyle_mixed_seed98401_1200.json',
        'results/sports_alias_bracket_multistyle_mixed_seed98402_1200.json',
        'results/corpus_joint_style_delimiter_audit_skip83500_n1000.json',
        'results/corpus_uppercase_repeat_joint_skip84500_n1000.json',
        'results/v14_zipf_seed99851_n480/v14_vs_fineweb_word_rank.json')]
    paths.append(HERE/'results/natural_sports_zipf_n480.json')
    paths.append(HERE/'results/v7_zipf_by_kind_n500.json')
    paths.extend(HERE/f'results/v7_action_edge_patch_seed{seed}.json'
                 for seed in (35051, 35052))
    paths.append(HERE/'results/v7_action_edge_patch_model34802.json')
    paths.extend(HERE/'results'/name for name in (
        'v7_action_rotary_pair_seed35151.json',
        'v7_action_rotary_pair_seed35251.json',
        'v7_action_rotary_pair_model34802.json',
        'v7_action_rotary_pair_model34802_rep.json'))
    paths.extend(HERE/'results'/name for name in (
        'v7_renderer_variant_rate025_n480.json',
        'v7_renderer_variant_rate05_n480.json',
        'v7_renderer_variant_n480.json'))
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
