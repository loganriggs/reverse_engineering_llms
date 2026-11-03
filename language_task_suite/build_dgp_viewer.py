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


def _rulebook():
    """Operational description of the V7 sampler shown in this viewer."""
    items = [
        {'title': 'Document mixture and training objective',
         'sampling': ('Each training document independently chooses narrative '
                      '25%, event report 30%, record listing 12%, score report '
                      '13%, or cross report 20%. The displayed samples are '
                      'fixed-seed draws from this same task-rich sampler; '
                      'these proportions are intentionally not estimates '
                      'of natural language.'),
         'rule': ('The V7 bilinear checkpoints were trained on 10,000 sampled '
                  'documents for 2,000 updates using ordinary next-token '
                  'cross-entropy at every token after the first. Padding is '
                  'ignored. The model must predict names, connective words, '
                  'punctuation, newlines, facts, and answer tokens; rule-site '
                  'labels are used only for evaluation. Longer documents '
                  'contribute more token losses. A site oracle is '
                  'available below, but a full exact next-token oracle for '
                  'every position exists only for the V14 narrative arm.')},
        {'title': 'Narrative: probabilistic local and distant rules',
         'sampling': ('Sample the V14 sentence stream to roughly 192 tokens '
                      'and stop at its last sentence boundary. Its explicit '
                      'rates include source questions 15%, scope questions '
                      '8%, brackets 6%, attribute facts 20%, and a 35% '
                      'chance to continue an attribute fact with “, so …”. '
                      'These are generator controls, not empirical text '
                      'frequencies.'),
         'rule': ('Local noun/verb and earlier-cue combinations set soft '
                  'next-token distributions. An earlier item-attribute fact '
                  'puts 80% mass on reusing its adjective at a later mention, '
                  'with the rest distributed over other adjectives; the '
                  'object has a similar 80% mass. In a local conditional '
                  'continuation, clean→uses, dusty→cleans, broken→repairs, '
                  'valuable→protects, rare→inspects, heavy→moves. Bracket '
                  'closers and other query answers depend on visible context. '
                  'Only the local action is strongly learned in the two V7 '
                  'checkpoints; harder binding must pass a behavioral gate.')},
        {'title': 'Event report: visible world state and report sources',
         'sampling': ('Sample objects with owner, color, place, aliases, and '
                      'a sequence of attributed updates. Choose among story, '
                      'summary, and log styles; intro/event/recap clauses '
                      'have 3/4/4 forms, with style entropy .7. Add asides '
                      'at rate .3, paragraph breaks at .22, and a header '
                      'at .5. Source questions are delayed past a different '
                      'report.'),
         'rule': ('A later alias query asks which object received a label. '
                  'Owner/color/place queries read the latest visible update '
                  'for that object. A source query asks who made the latest '
                  'report about a specified object and attribute—not merely '
                  'who spoke most recently. Sampled answers are verified '
                  'against a parser of visible text; V7 did not learn the '
                  'source rule reliably.')},
        {'title': 'Record listing: scoped key-value lookup',
         'sampling': ('Choose two distinct ships and two distinct category '
                      'codes, one room, and a query ship. Print the two '
                      'ship–room–category facts in random order. The '
                      'composition split withholds the queried ship–category '
                      'assignment while retaining its individual values.'),
         'rule': ('At “For the [ship], room [room] was category …”, return '
                  'the code in the earlier record matching both ship and '
                  'room. A room alone is insufficient.')},
        {'title': 'Score report: state update',
         'sampling': ('Choose two teams, initial scores from 1–5, and an '
                      'increment from 1–4 for the first team. A final-score '
                      'headline appears with probability .5. The query '
                      'names either team; a composition split withholds '
                      'selected initial-score × increment pairs.'),
         'rule': ('At the final score target, return the queried team’s '
                  'initial score plus its visible increment if it scored, '
                  'or its unchanged initial score otherwise. The optional '
                  'headline is a potential copy shortcut, so behavioral '
                  'tests must distinguish copying from updating.')},
        {'title': 'Cross report: composed retrieval',
         'sampling': ('Combine one event history, two source→reviewed-ship '
                      'assignments, and two ship–room–category records. '
                      'Shuffle Events, Reviewers, and Records section order. '
                      'The last event concerns a different object or '
                      'attribute from the query.'),
         'rule': ('At the final query, find the latest report source for the '
                  'specified object and attribute; map that source to its '
                  'reviewed ship; then look up that ship and room’s category. '
                  'The answer is independently recovered from the entire '
                  'visible prefix. Both tested V7 models failed this '
                  'behavioral and causal-edit gate, so no cross circuit '
                  'is claimed.')},
        {'title': 'Natural carrier: score copying and bracket closure',
         'sampling': ('Select a screened human RotoWire report. Insert a '
                      'validated final-score lead and a later score-copy '
                      'sentence; retain a natural parenthetical in between. '
                      'Independently choose base versus +1 winning score '
                      'and round versus square brackets. The all-token '
                      'training mixture draws 1,500 paired score reports '
                      'and 1,500 paired bracket reports; it never trains '
                      'on the joint 2×2 edited document. The carrier text '
                      'is sampled from a corpus, not generated by an exact '
                      'symbolic language model.'),
         'rule': ('At the score target, copy the higher final-score number '
                  'from the visible lead. At the bracket target, close the '
                  'latest unmatched visible opener. Changing the winning '
                  'score by one must change only the later score answer; '
                  'changing the opener must change only the closer. Both '
                  'rules are behaviorally learned in two small bilinear '
                  'models on heldout combined reports. Ordinary next-token '
                  'cross-entropy is trained at every nonpadding position '
                  'with extra weight at the active rule site; there is no '
                  'full oracle for the intervening human prose.')},
    ]
    sources = (
        'document_mixture_v7_cross.py; train_document_mixture_v7_cross.py',
        'document_mixture_v5.py; surface_mixture_v14_conditional.py',
        'document_world_v5_clause_realizations.py',
        'document_mixture_v5.py; document_mixture_v6.py',
        'document_mixture_v5.py; document_mixture_v6.py',
        'document_mixture_v7_cross.py',
        'sports_score_bracket_dgp.py; train_sports_score_bracket_mixture.py',
    )
    for item, source in zip(items, sources, strict=True):
        item['source_files'] = source.split('; ')
    return items


def _oracle_examples():
    """Actual fixed-seed prefixes and exact rule-site distributions."""
    choices = {'narrative': ('mixed_verb', 'mixed_object',
                             'attribute_adjective', 'conditional_action'),
               'event_report': ('source_attribution',),
               'record_listing': ('scoped_record',),
               'score_report': ('score_update',),
               'cross_report': ('cross_source_record',)}
    rows = []
    for index, kind in enumerate(KINDS):
        doc = sample_document(random.Random(98630+index), kind=kind)
        rendered, spans = _render_tokens(doc.tokens)
        for family in choices[kind]:
            site = next((s for s in doc.sites if s.family == family), None)
            if site is None:
                continue
            distribution = site_oracle(doc, site)
            assert abs(sum(distribution.values())-1) < 1e-8
            assert distribution.get(site.target, 0) > 0
            start = spans[site.index][0]
            ranked = sorted(distribution.items(), key=lambda item:
                            (-item[1], item[0]))
            rows.append({'kind': kind, 'family': family,
                         'prefix': rendered[max(0, start-170):start].lstrip(),
                         'prefix_clipped': start > 170,
                         'sampled': site.target,
                         'distribution': ranked,
                         'scope': ('Exact full-prefix generator marginal'
                                   if kind == 'narrative' else
                                   'Exact at this answer site; not an '
                                   'all-positions oracle')})
    return rows


def _latest_candidate():
    audit = _json(Path('results/sports_empirical_predicates_paired_n128.json'))
    fixed = _json(Path('results/sports_baseline_predicate_depth2_seed37301_s2000.json'))
    empirical = _json(Path('results/sports_empirical_predicate_depth2_seed37301_s2000.json'))
    panel = audit['arms']['rotowire_heldout']['form']
    return {'title': 'Separate sports-language surface candidate',
            'status': ('A typed bank of human-written player-stat predicates '
                       'improves form but remains far from human reports. '
                       'Its one-seed model learns the winner rule less '
                       'reliably at the same budget; margin and top-player '
                       'rules remain unlearned. It is not the V7 mixture '
                       'used for the rule-site distributions above.'),
            'report': 'EMPIRICAL_PLAYER_PREDICATE_PILOT.md',
            'rules': [
                {'title': 'Sampled game and player facts',
                 'sampling': ('Choose RotoWire or SBNation source games with '
                              'equal probability under a game-disjoint '
                              'train/held-out split. Sample home winner '
                              'with probability .55, high score 100–130, '
                              'and margin 1–25. Choose five distinct '
                              'roster players (three from winner, two '
                              'from loser) with unique point totals 8–32; '
                              'also sample rebounds and assists. Select '
                              'a score sentence compatible with the new '
                              'numbers.'),
                 'rule': ('The structured world fixes winner, margin, and '
                          'the highest-scoring player among the five. '
                          'Those three targets are checked independently '
                          'from visible text and under selective edits.'),
                 'source_files': ['sports_document_dgp.py']},
                {'title': 'Typed stat wording and discourse',
                 'sampling': ('The candidate arm samples one of 95 distinct '
                              'training-game-derived stat predicate forms '
                              'for each player, weighted by square root '
                              'of source frequency. Only points-first '
                              'forms with point/rebound/assist facts are '
                              'allowed. Five single-player clauses are '
                              'randomly ordered; a player clause leads '
                              'the score at rate .25. Recap sentence '
                              'wording and position vary, with paragraph '
                              'break rate .12.'),
                 'rule': ('Every rendered statistic must equal that '
                          'player’s sampled box-score fact. The three '
                          'recaps ask for winner, winning margin, and '
                          'which listed player has the most points. '
                          'This is not a full-document next-token oracle.'),
                 'source_files': ['sports_empirical_stat_renderer.py',
                                  'sports_document_dgp.py']},
                {'title': 'Model training and eligibility',
                 'sampling': ('Two depth-2, width-64, four-head bilinear '
                              'models received 4,000 sampled training '
                              'documents and 2,000 updates each, seed '
                              '37301, with either fixed or empirical '
                              'player-stat forms.'),
                 'rule': ('Both models used ordinary next-token '
                          'cross-entropy on every non-padding token '
                          'position, not just the three answer sites. '
                          'Those sites are evaluation labels only. '
                          'Under the empirical form, winner is partial '
                          'and margin/top-player remain unlearned, so '
                          'weight analysis of those rules is ineligible.'),
                 'source_files': ['train_sports_document_pilot.py']},
            ],
            'rows': [
                {'arm': label,
                 'top10_word_share': panel[label]['top10_word_share'],
                 'repeated_trigram_share': panel[label][
                     'repeated_across_docs_trigram_occurrence_share'],
                 'winner_accuracy': model['results']['2']['heldout']['site'][
                     'winner']['candidate_accuracy'] if model else None}
                for label, model in (('baseline', fixed),
                                     ('empirical', empirical),
                                     ('human_reference', None))]}


def _event_world_screen():
    pilot = _json(Path('results/sports_event_world_renderer_n100_form64_w30.json'))
    carrier = _json(Path('results/sports_event_natural_carrier_n64_w100.json'))
    metric = 'repeated_across_docs_trigram_occurrence_share'
    rows = []
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            key = source+'_'+group
            form = pilot['arms'][key]['form']
            natural = carrier['arms'][key]
            local = natural['local_repetitions']
            rows.append({
                'source': source, 'group': group,
                'eligible_games': natural['eligible_rows'],
                'generated_30': form['generated'][metric],
                'human_30': form['human_reference'][metric],
                'original_100': natural['paired_original'][metric],
                'insertion_100': natural['with_run_insertion'][metric],
                'inserted_window': sum(x['generated'] for x in local)/len(local),
                'human_run_window': sum(x['human_run'] for x in local)/len(local),
            })
    return {
        'status': ('Event-world arithmetic passes 800 visible run-number '
                   'sites, but both the fully sampled text pilot and the '
                   'natural-carrier insertion fail their human form screens. '
                   'Human retrospective reports rarely print both score '
                   'snapshots before a run number, so arithmetic completion '
                   'is a better target for a score-ledger genre. '
                   'Neither is an accepted training DGP; no model was '
                   'trained on these variants.'),
        'rows': rows,
        'examples': [pilot['arms'][source+'_train']['semantic']
                     ['generated_examples'][0]
                     for source in ('rotowire', 'sbnation')],
        'rules': [
            {'title': 'Empirical event world and split',
             'sampling': ('Sample a unique 2016/17 game aligned by date, '
                          'full teams, and final score to its box score '
                          'and play-by-play feed. Keep whole game IDs '
                          'apart across train and held-out partitions. '
                          'Carry final and all period-end scores, a top '
                          'scorer, and an event-verified regulation run.'),
             'rule': ('The run interval is a within-quarter scoreboard '
                      'change: one side scores 8–25, the other 1–12, '
                      'and the margin swings by at least six. Overtime '
                      'remains in game state but is excluded from this '
                      'first run renderer.'),
             'source_files': ['sports_event_world_dgp.py']},
            {'title': 'Visible run arithmetic',
             'sampling': ('Sample result, scorer, and period statements. '
                          'When a run is included, state two earlier '
                          'score snapshots and then an X–Y run claim. '
                          'Default run inclusion is 4% for RotoWire-style '
                          'documents and 22% for SBNation-style ones.'),
             'rule': ('The two run numbers must equal the named team’s '
                      'and opponent’s score changes between the visible '
                      'snapshots. Forced-run semantic audits passed 800 '
                      'number sites. Form remains far from human prose; '
                      'there is no all-token oracle or model learning '
                      'claim for this pilot.'),
             'source_files': ['sports_event_world_renderer.py',
                              'audit_sports_event_world_renderer.py']},
            {'title': 'Natural-carrier form control',
             'sampling': ('Insert the same event-verified run block into '
                          'an unchanged human report about the joined '
                          'game. Require a matching final score somewhere '
                          'in the report and at least 100 source tokens.'),
             'rule': ('The surrounding human prose is not generated and '
                      'has no complete symbolic oracle. Paired documents '
                      'look more natural, but the local run windows still '
                      'repeat far more than human run windows.'),
             'source_files': ['sports_event_natural_carrier.py',
                              'audit_sports_event_natural_carrier.py']},
            {'title': 'Observation regimes and next task choice',
             'sampling': ('In human retrospective reports, numeric run '
                          'sentences average about 32–35 tokens and '
                          'usually weave the run into other commentary. '
                          'Only a minority of screened run mentions have '
                          'even two score-shaped pairs in their '
                          'recent visible prefix.'),
             'rule': ('Do not assume the run number is computable from a '
                      'retrospective report prefix. A score-ledger or '
                      'live-report genre can expose score transitions '
                      'for arithmetic; retrospective prose instead '
                      'tests visible result, entity, and discourse '
                      'rules while run facts remain latent world facts.'),
             'source_files': ['audit_sports_run_syntax.py']},
        ]}


def _actplan_screen():
    audit = _json(Path('results/sports_document_actplan_v1_n100_form64.json'))
    ablation = _json(Path('results/sports_actplan_record_ablation_n64.json'))
    combined = _json(Path('results/sports_combined_result_record_n100_form64.json'))
    rows = []
    for key, arm in audit['arms'].items():
        form = arm['form']
        matched = ablation['arms'][key]['form']
        rows.append({
            'source': key.replace('_', ' '),
            'fixed_repeat': form['fixed_empirical'][
                'repeated_across_docs_trigram_occurrence_share'],
            'full_repeat': form['variable_actplan'][
                'repeated_across_docs_trigram_occurrence_share'],
            'human_repeat': form['human_reference'][
                'repeated_across_docs_trigram_occurrence_share'],
            'no_record_repeat': matched['no_records'][
                'repeated_across_docs_trigram_occurrence_share'],
            'matched_fixed_repeat': matched['fixed_empirical'][
                'repeated_across_docs_trigram_occurrence_share'],
        })
    combined_rows = []
    for group, arm in combined['arms'].items():
        panels = arm['form']['panels']
        combined_rows.append({
            'group': group,
            **{name: panels[name][
                'repeated_across_docs_trigram_occurrence_share']
                for name in ('fixed', 'variable_no_record',
                             'combined_record', 'human')},
        })
    return {'status': ('Newer act-plan V1 passes visible-answer and selective '
                       'edit checks, and samples which rule sites and facts '
                       'appear. Its default record sentences do not pass the '
                       'language-form gate: they increase phrase repetition. '
                       'A stricter RotoWire score-and-record lead avoids '
                       'most of that repetition penalty but remains far '
                       'from human form. No V1 model was trained or '
                       'interpreted.'),
            'report': 'SPORTS_DOCUMENT_ACTPLAN_V1_RESULT.md',
            'rows': rows,
            'combined_rows': combined_rows,
            'combined_report': 'SPORTS_COMBINED_RESULT_RECORD_RESULT.md'}


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
        {'name': 'Winning-score copy', 'rule': 'The higher score in an earlier '
         'final-result lead is repeated after intervening report prose.',
         'source': 'Natural sports carrier with controlled score edits',
         'example': '106–100 … The winning side scored 106 points.'},
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


def _lm_surface_screen():
    bank = _json(Path('results/sports_lm_surface_bank_candidates_seed37051.json'))
    return {'model': 'Qwen2.5-1.5B-Instruct',
            'rows': [{'fact': kind, 'attempts': item['attempts'],
                      'mechanical_accepts': item['accepted_candidates'],
                      'distinct': item['distinct_candidates']}
                     for kind, item in bank['kinds'].items()],
            'status': ('Rejected as a DGP source: the mechanical accepts '
                       'still include unsupported or awkward claims. '
                       'No bilinear model was trained on these forms.'),
            'report': 'LM_SURFACE_BANK_REJECTION_PILOT.md'}


def _recent_research():
    natural_score = _json(Path('results/natural_score_copy_mentions_screen.json'))
    score_transfer_a = _json(Path('results/sports_score_bracket_transfer_seed55801_n28.json'))
    score_transfer_b = _json(Path('results/sports_score_bracket_transfer_seed55802_n28.json'))
    score_ov_a = _json(Path('results/fit_score_bracket_ov_seed55801_rank8_both.json'))
    score_ov_b = _json(Path('results/fit_score_bracket_ov_seed55802_rank8_both.json'))
    score_overlap = _json(Path('results/score_rule_overlap_by_carrier_words.json'))
    pareto = _json(Path('results/fit_three_rule_bracket_ov_seed55302_rank2_bg0p3.json'))
    pareto_base = _json(Path('results/fit_three_rule_bracket_ov_seed55302_rank2_n80_n40.json'))
    bracket_fit_a = _json(Path('results/fit_three_rule_bracket_ov_seed55301_rank2_n80_n40.json'))
    bracket_fit_b = _json(Path('results/fit_three_rule_bracket_ov_seed55302_rank2_n80_n40.json'))
    three_ov_a = _json(Path('results/three_rule_ov_canonical_seed55301_n80_n40.json'))
    three_ov_b = _json(Path('results/three_rule_ov_canonical_seed55302_n80_n40.json'))
    three_a = _json(Path('results/sports_alias_bracket_caps_seed55301_s3000.json'))
    three_b = _json(Path('results/sports_alias_bracket_caps_seed55302_s6000.json'))
    variable_graph = _json(Path('results/v14_variable_graph_n128.json'))
    packed = _json(Path('results/v13_packed_intro_n64.json'))
    rule_windows = _json(Path('results/rule_window_form_v14_v2_n128.json'))
    agreement = _json(Path('results/treebank_agreement_subject_holdout_100kfit_50ktrain_balanced_depth2_4_3000.json'))
    causal = _json(Path('results/treebank_agreement_subject_holdout_100kfit_causal_n1000.json'))
    mixture = _json(Path('results/joint_natural_rules_seed41501_1000.json'))
    ov_a = _json(Path('results/sports_ov_canonical_kl_seed98201_n80_n40.json'))
    ov_b = _json(Path('results/sports_ov_canonical_kl_seed98202_n80_n40.json'))
    halftime = _json(Path('results/sports_halftime_clause_bank.json'))
    def rank8(data):
        return next(row for row in data['rows']
                    if row['method'] == 'data' and row['rank_per_head'] == 8)
    a4 = agreement['results']['4']['heldout']['site_pair_accuracy']
    c4 = causal['results'][next(key for key in causal['results']
                                if key.endswith('depth4'))]['counts']['attractor']
    v14_object = [rule_windows['synthetic'][f'v14_seed_{seed}']['form'][
        'mixed_object']['repeated_across_docs_trigram_occurrence_share']
        for seed in (0, 1)]
    return [
        {'title': 'Natural score-copy queries are rare',
         'status': 'Corpus support and semantic validity limit the next DGP',
         'finding': ("Among "
                     f"{natural_score['groups']['train']['validated_result_lead']} "
                     "validated training report leads, only "
                     f"{natural_score['groups']['train']['later_high_near_points']} "
                     "later mention the winning score near ‘points’ without "
                     "the losing score; the heldout count is "
                     f"{natural_score['groups']['heldout']['later_high_near_points']}. "
                     "Manual inspection finds some opponent or part-of-total "
                     "meanings, so these are upper bounds, not safe queries."),
         'report': 'SCORE_COPY_NATURAL_MENTION_SCREEN.md'},
        {'title': 'Two rules compose in heldout reports',
         'status': 'Both learned without jointly edited training reports',
         'finding': ("On the same 28-game heldout joint panel, two bilinear "
                     "models score paired bracket/score-copy accuracy "
                     f"{100*score_transfer_a['score']['bracket']['paired_top1']:.1f}%/"
                     f"{100*score_transfer_a['score']['numeric']['paired_top1']:.1f}% "
                     "and "
                     f"{100*score_transfer_b['score']['bracket']['paired_top1']:.1f}%/"
                     f"{100*score_transfer_b['score']['numeric']['paired_top1']:.1f}%. "
                     "The score lead/query is inserted around natural prose, "
                     "so its form remains an open realism limit."),
         'report': 'SCORE_BRACKET_COMPOSITION_RESULT.md'},
        {'title': 'Low-rank OV needs both rule objectives',
         'status': 'Separate-rule fitting transfers to their heldout combination',
         'finding': ("At rank eight per head, fitting OV factors on separate "
                     "score and bracket sites gives joint paired accuracy "
                     f"{100*score_ov_a['final']['bracket']['paired_top1']:.1f}%/"
                     f"{100*score_ov_a['final']['numeric']['paired_top1']:.1f}% "
                     "and "
                     f"{100*score_ov_b['final']['bracket']['paired_top1']:.1f}%/"
                     f"{100*score_ov_b['final']['numeric']['paired_top1']:.1f}%. "
                     "Fitting only one site generally preserves that rule "
                     "better than the other."),
         'report': 'SCORE_BRACKET_OV_TRANSFER_RESULT.md'},
        {'title': 'Score arithmetic can pair with brackets, but not all three rules',
         'status': 'Natural report overlap screens the next DGP pilot',
         'finding': ("At 100 carrier words, conservative score reports with "
                     "a retained bracket site number "
                     f"{score_overlap['by_carrier_words']['100']['train']['score_and_bracket']} "
                     "training and "
                     f"{score_overlap['by_carrier_words']['100']['heldout']['score_and_bracket']} "
                     "heldout games. Adding the surname rule leaves only "
                     f"{score_overlap['by_carrier_words']['100']['train']['all_three']}/"
                     f"{score_overlap['by_carrier_words']['100']['heldout']['all_three']}; "
                     "that intersection is too small for a three-rule model."),
         'report': 'SCORE_RULE_OVERLAP_SCREEN.md'},
        {'title': 'Bracket fidelity versus ordinary language fidelity',
         'status': 'Moderate background protection improves the tradeoff',
         'finding': ("At rank two per head, adding background KL with "
                     "weight 0.3 changes heldout bracket paired accuracy "
                     f"from {100*pareto_base['final']['bracket']['paired_top1']:.1f}% "
                     f"to {100*pareto['final']['bracket']['paired_top1']:.1f}% "
                     "while lowering background KL from "
                     f"{pareto_base['final_teacher_kl']['background']:.2f} "
                     f"to {pareto['final_teacher_kl']['background']:.2f} "
                     "on one checkpoint. The second checkpoint shows the "
                     "same tradeoff direction."),
         'report': 'BRACKET_OV_FIDELITY_PARETO_RESULT.md'},
        {'title': 'Rule-aware low-rank fit rescues bracket behavior',
         'status': 'Heldout rule fidelity recovers; background KL worsens',
         'finding': ("At two value channels per head, fitting only OV factors "
                     "to teacher bracket-site predictions raises heldout "
                     "paired bracket accuracy from "
                     f"{100*bracket_fit_a['initial']['bracket']['paired_top1']:.1f}% "
                     f"to {100*bracket_fit_a['final']['bracket']['paired_top1']:.1f}% "
                     "and from "
                     f"{100*bracket_fit_b['initial']['bracket']['paired_top1']:.1f}% "
                     f"to {100*bracket_fit_b['final']['bracket']['paired_top1']:.1f}%. "
                     "This uses supervised task sites and trades off ordinary "
                     "background predictions."),
         'report': 'BRACKET_AWARE_OV_DISTILLATION_RESULT.md'},
        {'title': 'Average activation fidelity misses a learned rule',
         'status': 'Weight-only rank-4 OV truncation preserves bracket better',
         'finding': ("At four value channels per head, weight-only versus "
                     "activation-aware reduction retains "
                     f"{100*next(r for r in three_ov_a['rows'] if r['method']=='weight' and r['rank_per_head']==4)['heldout']['bracket']['paired_top1']:.1f}%/"
                     f"{100*next(r for r in three_ov_a['rows'] if r['method']=='data' and r['rank_per_head']==4)['heldout']['bracket']['paired_top1']:.1f}% "
                     "paired bracket answers in one checkpoint, and "
                     f"{100*next(r for r in three_ov_b['rows'] if r['method']=='weight' and r['rank_per_head']==4)['heldout']['bracket']['paired_top1']:.1f}%/"
                     f"{100*next(r for r in three_ov_b['rows'] if r['method']=='data' and r['rank_per_head']==4)['heldout']['bracket']['paired_top1']:.1f}% "
                     "in the other, despite lower average reconstruction error "
                     "from the activation-aware method."),
         'report': 'THREE_RULE_OV_REDUCTION_RESULT.md'},
        {'title': 'Three rules in the same human report',
         'status': 'Bracket learned; name reuse weak; case has a word-frequency shortcut',
         'finding': (f"Two bilinear runs reach {100*three_a['test']['bracket']['paired_top1']:.1f}%/"
                     f"{100*three_b['test']['bracket']['paired_top1']:.1f}% heldout paired bracket accuracy, "
                     f"but name reuse only {100*three_a['test']['alias']['paired_top1']:.1f}%/"
                     f"{100*three_b['test']['alias']['paired_top1']:.1f}%. "
                     "Capitalization responds to punctuation, yet mostly predicts The/the "
                     "for rarer words. Treat only bracket as a positive circuit control."),
         'report': 'SPORTS_THREE_RULE_NATURAL_PILOT.md'},
        {'title': 'Larger sampled event graphs increase rule sites, not form',
         'status': 'Graph controls work; language-form gate fails',
         'finding': ("Variable 2–6-entity graphs average more scored sites "
                     "than fixed three-entity controls, but first-100-word "
                     "repeated trigrams rise from "
                     f"{100*variable_graph['panels']['seed_0']['fixed_3']['first100_form']['repeated_across_docs_trigram_occurrence_share']:.1f}% "
                     "to "
                     f"{100*variable_graph['panels']['seed_0']['variable']['first100_form']['repeated_across_docs_trigram_occurrence_share']:.1f}% "
                     "in one 128-document panel, with the same direction "
                     "in the second."),
         'report': 'VARIABLE_PROPOSITION_GRAPH_RESULT.md'},
        {'title': 'Packing facts across clauses does not solve form',
         'status': 'Truth checks pass; paired V10 form comparison fails',
         'finding': (f"Across two 64-world panels, full packing has "
                     f"{100*packed['panels'][0]['first100_form']['pack_1']['repeated_across_docs_trigram_occurrence_share']:.1f}%/"
                     f"{100*packed['panels'][1]['first100_form']['pack_1']['repeated_across_docs_trigram_occurrence_share']:.1f}% "
                     "cross-document repeated trigrams versus "
                     f"{100*packed['panels'][0]['first100_form']['v10']['repeated_across_docs_trigram_occurrence_share']:.1f}%/"
                     f"{100*packed['panels'][1]['first100_form']['v10']['repeated_across_docs_trigram_occurrence_share']:.1f}% "
                     "for V10. The proposition graph itself needs variation."),
         'report': 'PACKED_PROPOSITION_INTRO_RESULT.md'},
        {'title': 'Rule-bearing language form remains repetitive',
         'status': 'V14 and five-frame V2 fail the local form gate',
         'finding': (f'V14 mixed-object contexts repeat cross-document '
                     f'trigrams at {100*v14_object[0]:.1f}%/'
                     f'{100*v14_object[1]:.1f}% across two seeds, versus '
                     'roughly 1% in matched-length FineWeb windows. '
                     'V2 is more repetitive; it remains an oracle fixture.'),
         'report': 'RULE_WINDOW_FORM_AUDIT.md'},
        {'title': 'Agreement needs whole-subject holdouts',
         'status': 'Partial rule learning; no syntax circuit claim',
         'finding': f"Depth 4 gets {100*a4['plain']['accuracy']:.1f}% on plain subjects but {100*a4['attractor']['accuracy']:.1f}% with an opposite-number noun. Only {100*c4['both_causal']/c4['n']:.1f}% of heldout attractor edit triples show both required responses.",
         'report': 'TREEBANK_AGREEMENT_HOLDOUT_RESULT.md'},
        {'title': 'Three-rule natural-text mixture',
         'status': 'Bracket learned; alias partial; spelling unlearned',
         'finding': f"The mixed model keeps {100*mixture['arms']['mixed']['test']['sports']['bracket']['paired_top1']:.1f}% heldout paired bracket accuracy. Spelling paired correctness is {100*mixture['arms']['mixed']['test']['style']['both_two_way']:.1f}% despite lower FineWeb token loss.",
         'report': 'JOINT_NATURAL_RULE_MIXTURE_PILOT.md'},
        {'title': 'OV gauge and truncation',
         'status': 'Exact full-rank gauge; rule-specific compression',
         'finding': f"Activation-weighted rank 8 per attention head keeps bracket paired accuracy at {100*rank8(ov_a)['heldout']['bracket']['paired_top1']:.1f}% and {100*rank8(ov_b)['heldout']['bracket']['paired_top1']:.1f}% in two seeds, but alias fidelity degrades.",
         'report': 'SPORTS_OV_CANONICAL_TRUNCATION.md'},
        {'title': 'Halftime-to-final rule candidate',
         'status': 'Natural opportunity, but sentence bank too small',
         'finding': f"A conservative full-sentence semantic validator accepts {halftime['sbnation/train']['counts']['accepted_clauses']} SBNation training clauses and {halftime['sbnation/heldout']['counts']['accepted_clauses']} heldout clause. Surface matches alone are not safe training labels.",
         'report': 'HALFTIME_CLAUSE_VALIDATOR_RESULT.md'},
    ]


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
            'research_updates': _recent_research(),
            'rulebook': _rulebook(),
            'oracle_examples': _oracle_examples(),
            'latest_candidate': _latest_candidate(),
            'event_world_screen': _event_world_screen(),
            'actplan_screen': _actplan_screen(),
            'zipf': _zipf_comparison(),
            'action_path': _action_path_summary(),
            'pair_reduction': _pair_reduction_summary(),
            'lm_surface_screen': _lm_surface_screen(),
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


def write_data(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text('window.DGP_VIEWER_DATA = '+
                         json.dumps(data, ensure_ascii=False).replace(
                             '</', '<\\/')+';\n')
    os.replace(temporary, path)
    print('Updated', path, 'with', len(data['examples']),
          'sampled examples at', data['generated_at'], flush=True)


def write_public_html(path, data):
    """Publish only generated examples and aggregate audits, never source prose."""
    public = dict(data)
    public['examples'] = [example for example in data['examples']
                          if example['group'] == 'symbolic']
    public['public_snapshot'] = True
    public['data_policy'] = ('Public snapshot: generated examples and aggregate '
                             'audits only. Natural-source excerpts stay local.')
    html = (HERE/'dgp_viewer.html').read_text()
    marker = '<script src="dgp_viewer_data.js"></script>'
    assert html.count(marker) == 1
    payload = json.dumps(public, ensure_ascii=False).replace('</', '<\\/')
    html = html.replace(marker, '<script>window.DGP_VIEWER_DATA = '+payload+';</script>')
    for example in data['examples']:
        if example['group'] == 'natural':
            for arm in example['arms']:
                assert arm['text'] not in html, 'Natural source text in public HTML'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(html)
    os.replace(temporary, path)
    print('Updated public snapshot', path, 'with', len(public['examples']),
          'generated examples', flush=True)


def _write_both(local_path, public_path, data):
    write_data(local_path, data)
    write_public_html(public_path, data)
    pages_path = HERE.parent/'docs'/'index.html'
    if public_path.resolve() != pages_path.resolve():
        write_public_html(pages_path, data)


def _fingerprint():
    paths = [HERE/name for name in (
        'build_dgp_viewer.py', 'dgp_viewer.html',
        'document_mixture_v7_cross.py',
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
        'results/sports_alias_bracket_caps_seed55301_s3000.json',
        'results/sports_alias_bracket_caps_seed55302_s6000.json',
        'results/three_rule_ov_canonical_seed55301_n80_n40.json',
        'results/three_rule_ov_canonical_seed55302_n80_n40.json',
        'results/fit_three_rule_bracket_ov_seed55301_rank2_n80_n40.json',
        'results/fit_three_rule_bracket_ov_seed55302_rank2_n80_n40.json',
        'results/fit_three_rule_bracket_ov_seed55302_rank2_bg0p3.json',
        'results/score_rule_overlap_by_carrier_words.json',
        'results/sports_score_bracket_transfer_seed55801_n28.json',
        'results/sports_score_bracket_transfer_seed55802_n28.json',
        'results/fit_score_bracket_ov_seed55801_rank8_both.json',
        'results/fit_score_bracket_ov_seed55802_rank8_both.json',
        'results/natural_score_copy_mentions_screen.json',
        'results/sports_alias_bracket_multistyle_mixed_seed98401_1200.json',
        'results/sports_alias_bracket_multistyle_mixed_seed98402_1200.json',
        'results/corpus_joint_style_delimiter_audit_skip83500_n1000.json',
        'results/corpus_uppercase_repeat_joint_skip84500_n1000.json',
        'results/v14_zipf_seed99851_n480/v14_vs_fineweb_word_rank.json')]
    paths.extend(HERE/'results'/name for name in (
        'treebank_agreement_subject_holdout_100kfit_50ktrain_balanced_depth2_4_3000.json',
        'treebank_agreement_subject_holdout_100kfit_causal_n1000.json',
        'joint_natural_rules_seed41501_1000.json',
        'sports_ov_canonical_kl_seed98201_n80_n40.json',
        'sports_ov_canonical_kl_seed98202_n80_n40.json',
        'sports_halftime_clause_bank.json'))
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
    paths.append(HERE/'results/sports_lm_surface_bank_candidates_seed37051.json')
    paths.extend(HERE/'results'/name for name in (
        'sports_empirical_predicates_paired_n128.json',
        'sports_baseline_predicate_depth2_seed37301_s2000.json',
        'sports_empirical_predicate_depth2_seed37301_s2000.json'))
    paths.extend(HERE/'results'/name for name in (
        'sports_event_world_renderer_n100_form64_w30.json',
        'sports_event_natural_carrier_n64_w100.json',
        'sports_run_syntax_2016.json',
        'sports_document_actplan_v1_n100_form64.json',
        'sports_actplan_record_ablation_n64.json',
        'sports_combined_result_record_n100_form64.json'))
    paths.append(HERE/'sports_document_actplan_v1.py')
    paths.append(HERE/'sports_combined_result_record_renderer.py')
    paths.append(HERE/'results/rule_window_form_v14_v2_n128.json')
    paths.append(HERE/'results/v13_packed_intro_n64.json')
    paths.append(HERE/'results/v14_variable_graph_n128.json')
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
    p.add_argument('--public-html', type=Path,
                   default=HERE/'dgp_viewer_iphone.html')
    p.add_argument('--cache-path', type=Path, default=DEFAULT_CACHE)
    p.add_argument('--refresh-corpus', action='store_true')
    p.add_argument('--watch', action='store_true',
                   help='Rebuild when the DGP code or result files change')
    args = p.parse_args()
    _write_both(args.output, args.public_html,
                build(args.cache_path, args.refresh_corpus))
    if args.watch:
        fingerprint = _fingerprint()
        while True:
            time.sleep(15)
            current = _fingerprint()
            if current != fingerprint:
                _write_both(args.output, args.public_html,
                            build(args.cache_path, False))
                fingerprint = current
    sys.stdout.flush()
    os._exit(0)  # streaming datasets may crash during interpreter shutdown


if __name__ == '__main__':
    main()
