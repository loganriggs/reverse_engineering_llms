"""Evaluate v6 source answers and supported source-binding interventions."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from eval_surface_mixture_edits import predict, score
from eval_surface_mixture_v4_edits import full_support
from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v6 import (SPEAKERS, SurfaceMixtureV6,
                                source_pair_heldout)


def statements(history):
    out, offset = [], 0
    for raw in history:
        clause = _content(raw)
        if clause and clause[0] == 'As':
            start = offset+len(raw)-len(clause)
            quoted = clause[4] == '"'
            obj = clause[6] if quoted else clause[5]
            out.append(dict(object=obj, speaker=clause[1],
                            speaker_index=start+1, start=start,
                            quoted=quoted))
        offset += len(raw)
    return out


def build(gen, n, length):
    ids, labels, _ = gen.sample_batch(n, length+1)
    questions, pairs = [], []
    attempted = Counter()
    for row, labs in zip(ids, labels):
        words = [gen.id2token[i] for i in row]
        for t in range(1, length+1):
            if labs[t] != 'source_answer':
                continue
            prefix = words[:t]
            history, raw = _split(prefix)
            clause = _content(raw)
            prior_statements = statements(history)
            source = {item['object']: item['speaker']
                      for item in prior_statements}
            key = clause[3]
            target = source[key]
            queried_quoted = next(item['quoted'] for item in prior_statements
                                  if item['object'] == key)
            pair_holdout = source_pair_heldout(target, key)
            assert gen.next_distribution(prefix) == {target: 1.}
            masses = Counter(source.values())
            prior = [count/len(source) for count in masses.values()]
            modal = {speaker for speaker, count in masses.items()
                     if count == max(masses.values())}
            questions.append(dict(prefix=row[:t], target=gen.token2id[target],
                                  pair_holdout=pair_holdout,
                                  queried_quoted=queried_quoted,
                                  key_blind_bayes=max(prior),
                                  key_blind_entropy=-sum(p*math.log(p) for p in prior),
                                  latest_speaker_correct=(
                                      prior_statements[-1]['speaker'] == target),
                                  first_speaker_correct=(
                                      prior_statements[0]['speaker'] == target),
                                  latest_speaker=gen.token2id[
                                      prior_statements[-1]['speaker']],
                                  first_speaker=gen.token2id[
                                      prior_statements[0]['speaker']],
                                  modal_speakers=[gen.token2id[s] for s in modal]))
            # The query key is six tokens before the source-answer slot.
            query_index = t-6
            attempted['query_object'] += 1
            for other, speaker in source.items():
                if speaker == target:
                    continue
                edited = prefix.copy()
                edited[query_index] = other
                if full_support(gen, edited):
                    q1 = gen.next_distribution(edited)
                    assert q1 == {speaker: 1.}
                    pairs.append(dict(kind='query_object',
                                      pair_holdout=pair_holdout,
                                      queried_quoted=queried_quoted,
                                      original=[gen.token2id[w] for w in prefix],
                                      edited=[gen.token2id[w] for w in edited],
                                      q_before={target: 1.}, q_after=q1))
                    break
            # Change the source of the queried quote. Full support also checks
            # subsequent questions whose eligibility may depend on speakers.
            relevant_rank, relevant = next((rank, item)
                                           for rank, item in enumerate(prior_statements)
                                           if item['object'] == key)
            attempted['statement_speaker'] += 1
            for candidate in SPEAKERS:
                if candidate == target:
                    continue
                edited = prefix.copy()
                edited[relevant['speaker_index']] = candidate
                if not full_support(gen, edited):
                    continue
                q1 = gen.next_distribution(edited)
                if q1 == {candidate: 1.}:
                    pairs.append(dict(kind='statement_speaker',
                                      pair_holdout=pair_holdout,
                                      queried_quoted=queried_quoted,
                                      quote_rank=relevant_rank,
                                      n_statements=len(prior_statements),
                                      original=[gen.token2id[w] for w in prefix],
                                      edited=[gen.token2id[w] for w in edited],
                                      q_before={target: 1.}, q_after=q1))
                    break
            if 0 < gen.unquoted_source_rate < 1:
                attempted['statement_form'] += 1
                edited = prefix.copy()
                start = relevant['start']
                if relevant['quoted']:
                    del edited[start+9]
                    del edited[start+4]
                else:
                    edited.insert(start+8, '"')
                    edited.insert(start+4, '"')
                if full_support(gen, edited):
                    q1 = gen.next_distribution(edited)
                    assert q1 == {target: 1.}
                    pairs.append(dict(kind='statement_form',
                                      pair_holdout=pair_holdout,
                                      queried_quoted=queried_quoted,
                                      quote_rank=relevant_rank,
                                      n_statements=len(prior_statements),
                                      original=[gen.token2id[w] for w in prefix],
                                      edited=[gen.token2id[w] for w in edited],
                                      q_before={target: 1.}, q_after=q1))
            # Another quotation's speaker must leave this query unchanged.
            attempted['other_speaker'] += 1
            for other_rank, item in enumerate(prior_statements):
                if item['object'] == key:
                    continue
                for candidate in SPEAKERS:
                    if candidate == item['speaker']:
                        continue
                    edited = prefix.copy()
                    edited[item['speaker_index']] = candidate
                    if full_support(gen, edited):
                        q1 = gen.next_distribution(edited)
                        assert q1 == {target: 1.}
                        pairs.append(dict(kind='other_speaker',
                                          pair_holdout=pair_holdout,
                                          queried_quoted=queried_quoted,
                                          quote_rank=other_rank,
                                          n_statements=len(prior_statements),
                                          original=[gen.token2id[w] for w in prefix],
                                          edited=[gen.token2id[w] for w in edited],
                                          q_before={target: 1.}, q_after=q1))
                        break
                else:
                    continue
                break
    return questions, pairs, attempted


def prediction(items, probs):
    target = np.array([item['target'] for item in items])
    correct = probs[np.arange(len(items)), target]
    predicted = probs.argmax(-1)
    return dict(n=len(items), accuracy=float(np.mean(probs.argmax(-1) == target)),
                answer_ce=float(np.mean(-np.log(np.maximum(correct, 1e-30)))),
                correct_token_probability=float(np.mean(correct)),
                key_blind_bayes_accuracy=float(np.mean(
                    [item['key_blind_bayes'] for item in items])),
                key_blind_answer_ce=float(np.mean(
                    [item['key_blind_entropy'] for item in items])),
                latest_speaker_accuracy=float(np.mean(
                    [item['latest_speaker_correct'] for item in items])),
                first_speaker_accuracy=float(np.mean(
                    [item['first_speaker_correct'] for item in items])),
                model_predicts_latest_speaker=float(np.mean([
                    predicted[i] == item['latest_speaker']
                    for i, item in enumerate(items)])),
                model_predicts_first_speaker=float(np.mean([
                    predicted[i] == item['first_speaker']
                    for i, item in enumerate(items)])),
                model_predicts_modal_speaker=float(np.mean([
                    predicted[i] in item['modal_speakers']
                    for i, item in enumerate(items)])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dirs', nargs='+', required=True)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=9101)
    parser.add_argument('--quote-rate', type=float, default=.30)
    parser.add_argument('--source-rate', type=float, default=.30)
    parser.add_argument('--question-rate', type=float, default=.05)
    parser.add_argument('--unquoted-source-rate', type=float, default=0.)
    parser.add_argument('--checkpoint-steps', default='',
                        help='Comma-separated saved development steps; empty evaluates final models')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    gen = SurfaceMixtureV6(seed=args.seed, quote_rate=args.quote_rate,
                           source_rate=args.source_rate,
                           question_rate=args.question_rate,
                           unquoted_source_rate=args.unquoted_source_rate,
                           counterbalanced_queries=True)
    questions, pairs, attempted = build(gen, args.n, args.length)
    assert questions and pairs
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    results = {}
    for directory in args.model_dirs:
        folder = Path(directory)
        saved = json.loads((folder/'results.json').read_text())
        records = (saved['records'] if 'records' in saved else
                   [dict(depth=2, model_config=saved['model_config'])])
        for record in records:
            depth = record['depth']
            steps = ([int(step) for step in args.checkpoint_steps.split(',')]
                     if args.checkpoint_steps else [None])
            for step in steps:
                checkpoint = (folder/f'model_depth{depth}_step{step}.pt'
                              if step is not None else
                              folder/f'model_depth{depth}.pt')
                model = BilinearGPT(Config(**record['model_config']))
                model.load_state_dict(torch.load(checkpoint, map_location='cpu',
                                                weights_only=True))
                model.eval()
                probs = predict(model, [item['prefix'] for item in questions],
                                len(gen.vocab))
                before = predict(model, [pair['original'] for pair in pairs],
                                 len(gen.vocab))
                after = predict(model, [pair['edited'] for pair in pairs],
                                len(gen.vocab))
                name = (f'{folder.name}_depth{depth}' +
                        (f'_step{step}' if step is not None else ''))
                rank_groups = {}
                for kind in ('statement_speaker', 'other_speaker'):
                    for rank_name, predicate in (
                        ('first', lambda pair: pair['quote_rank'] == 0),
                        ('last', lambda pair: pair['quote_rank'] ==
                         pair['n_statements']-1),
                        ('middle', lambda pair: 0 < pair['quote_rank'] <
                         pair['n_statements']-1)):
                        indices = [i for i, pair in enumerate(pairs)
                                   if pair['kind'] == kind and predicate(pair)]
                        if indices:
                            rank_groups[f'{kind}_{rank_name}'] = score(
                                [pairs[i] for i in indices], before[indices],
                                after[indices], gen)[kind]
                results[name] = dict(prediction=prediction(questions, probs),
                                     edits=score(pairs, before, after, gen),
                                     edit_by_quote_rank=rank_groups)
                held_question_ix = [i for i, item in enumerate(questions)
                                    if item['pair_holdout']]
                held_pair_ix = [i for i, item in enumerate(pairs)
                                if item['pair_holdout']]
                if held_question_ix and held_pair_ix:
                    results[name]['pair_holdout'] = dict(
                        prediction=prediction(
                            [questions[i] for i in held_question_ix],
                            probs[held_question_ix]),
                        edits=score([pairs[i] for i in held_pair_ix],
                                    before[held_pair_ix], after[held_pair_ix], gen),
                        n_questions=len(held_question_ix),
                        n_pairs=len(held_pair_ix))
                for form_name, quoted in (('quoted', True),
                                          ('unquoted', False)):
                    question_ix = [i for i, item in enumerate(questions)
                                   if item['queried_quoted'] == quoted]
                    pair_ix = [i for i, item in enumerate(pairs)
                               if item['queried_quoted'] == quoted]
                    if question_ix and pair_ix:
                        results[name].setdefault('by_source_form', {})[form_name] = dict(
                            prediction=prediction([questions[i] for i in question_ix],
                                                  probs[question_ix]),
                            edits=score([pairs[i] for i in pair_ix],
                                        before[pair_ix], after[pair_ix], gen))
                        held_question_form = [i for i in question_ix
                                              if questions[i]['pair_holdout']]
                        if held_question_form:
                            results[name]['by_source_form'][form_name][
                                'pair_holdout_prediction'] = prediction(
                                    [questions[i] for i in held_question_form],
                                    probs[held_question_form])
                print(name, results[name], flush=True)
    output = dict(config={**vars(args), 'output': str(args.output)},
                  n_questions=len(questions),
                  attempted=dict(attempted),
                  pair_counts=dict(Counter(pair['kind'] for pair in pairs)),
                  results=results,
                  caveat='Every source answer and edit is checked against the '
                         'exact visible-prefix oracle. Query-object and source '
                         'speaker edits change the answer; other-speaker edits '
                         'preserve it. Accuracy, oracle effects, and model effects '
                         'are distinct measurements.')
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
