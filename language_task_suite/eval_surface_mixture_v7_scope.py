"""Evaluate section-key competence and exact supported edits for v7."""

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
from surface_mixture_v7 import COLORS, SurfaceMixtureV7
from surface_mixture_v8 import SurfaceMixtureV8
from softmax_attention_control import configure_attention


def fact_entries(history):
    entries = {}
    active = None
    offset = 0
    for raw in history:
        clause = _content(raw)
        start = offset + len(raw) - len(clause)
        if clause and clause[0] == 'Section':
            active = clause[1]
        elif clause and clause[0] == 'Detail':
            assert active is not None
            entries[active] = dict(color=clause[5], index=start+5)
            active = None
        offset += len(raw)
    return entries


def build(gen, n, length, max_questions):
    ids, labels, _ = gen.sample_batch(n, length+1)
    questions, pairs = [], []
    attempted = Counter()
    for row, labs in zip(ids, labels):
        words = [gen.id2token[i] for i in row]
        for t in range(1, length+1):
            if labs[t] != 'scope_answer':
                continue
            prefix = words[:t]
            history, raw = _split(prefix)
            key = _content(raw)[7]
            facts = fact_entries(history)
            target = facts[key]['color']
            assert gen.next_distribution(prefix) == {target: 1.}
            questions.append(dict(prefix=row[:t], target=gen.token2id[target],
                                  key_blind_bayes=1/len(facts),
                                  key_blind_ce=math.log(len(facts))))
            attempted['section_key'] += 1
            key_index = t-4
            for other, fact in facts.items():
                if other == key:
                    continue
                edited = prefix.copy()
                edited[key_index] = other
                if full_support(gen, edited):
                    assert fact['color'] != target
                    pairs.append(dict(kind='section_key',
                                      original=[gen.token2id[w] for w in prefix],
                                      edited=[gen.token2id[w] for w in edited],
                                      q_before={target: 1.},
                                      q_after={fact['color']: 1.}))
                    break
            for kind, section in (('relevant_color', key),
                                  ('irrelevant_color', next(
                                      other for other in facts if other != key))):
                attempted[kind] += 1
                old = facts[section]['color']
                for color in COLORS:
                    if color == old:
                        continue
                    edited = prefix.copy()
                    edited[facts[section]['index']] = color
                    if not full_support(gen, edited):
                        continue
                    q_after = gen.next_distribution(edited)
                    if kind == 'relevant_color' and q_after != {color: 1.}:
                        continue
                    if kind == 'irrelevant_color' and q_after != {target: 1.}:
                        continue
                    pairs.append(dict(kind=kind,
                                      original=[gen.token2id[w] for w in prefix],
                                      edited=[gen.token2id[w] for w in edited],
                                      q_before={target: 1.}, q_after=q_after))
                    break
            if len(questions) >= max_questions:
                return questions, pairs, attempted
    return questions, pairs, attempted


def prediction(questions, probs):
    targets = np.array([item['target'] for item in questions])
    return dict(n=len(questions),
                accuracy=float(np.mean(probs.argmax(-1) == targets)),
                answer_ce=float(np.mean(-np.log(np.maximum(
                    probs[np.arange(len(targets)), targets], 1e-30)))),
                key_blind_bayes_accuracy=float(np.mean([
                    item['key_blind_bayes'] for item in questions])),
                key_blind_answer_ce=float(np.mean([
                    item['key_blind_ce'] for item in questions])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--n', type=int, default=600)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--max-questions', type=int, default=250)
    p.add_argument('--seed', type=int, default=9101)
    p.add_argument('--torch-threads', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.torch_threads < 1:
        p.error('--torch-threads must be positive')
    torch.set_num_threads(a.torch_threads)
    saved = json.loads((a.model_dir/'results.json').read_text())
    generator_class = {'v7': SurfaceMixtureV7,
                       'v8': SurfaceMixtureV8}[saved['config']['generator']]
    gen = generator_class(seed=a.seed, **saved['generator_kwargs'])
    questions, pairs, attempted = build(gen, a.n, a.length, a.max_questions)
    assert questions and pairs
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    results = {}
    for record in saved['records']:
        depth = record['depth']
        model = configure_attention(BilinearGPT(Config(**record['model_config'])),
                                    saved['config'].get('attention', 'bilinear'))
        model.load_state_dict(torch.load(a.model_dir/f'model_depth{depth}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        q_probs = predict(model, [item['prefix'] for item in questions], len(gen.vocab))
        before = predict(model, [pair['original'] for pair in pairs], len(gen.vocab))
        after = predict(model, [pair['edited'] for pair in pairs], len(gen.vocab))
        results[f'depth{depth}'] = dict(prediction=prediction(questions, q_probs),
                                       edits=score(pairs, before, after, gen))
        print(depth, results[f'depth{depth}'], flush=True)
    output = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  n_questions=len(questions), attempted=dict(attempted),
                  pair_counts=dict(Counter(pair['kind'] for pair in pairs)),
                  results=results,
                  caveat='All edits pass exact full-prefix support. The '
                         'key and relevant-color edits change the oracle '
                         'answer; irrelevant-color edits leave it unchanged.')
    a.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
