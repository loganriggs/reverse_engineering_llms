"""Exact-oracle causal tests of learned numeric comparison."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from audit_numeric_report import edit_pairs
from eval_surface_mixture_edits import predict
from numeric_report_language import (ID2TOKEN, NumericReportLanguage,
                                     TOKEN2ID, VOCAB)


def build_pairs(nstreams, length, seed):
    gen = NumericReportLanguage(seed=seed)
    rows = []
    for _ in range(nstreams):
        ids, labels, _ = gen.sample_sequence(length)
        words = [ID2TOKEN[i] for i in ids]
        for at, label in enumerate(labels):
            if label != 'score_comparison_answer':
                continue
            original = words[:at]
            q0, _ = gen.next_distribution(original)
            for kind, edited in edit_pairs(words, at, gen).items():
                q1, _ = gen.next_distribution(edited)
                rows.append(dict(kind=kind,
                                 original=[TOKEN2ID[word] for word in original],
                                 edited=[TOKEN2ID[word] for word in edited],
                                 oracle_before=next(iter(q0)),
                                 oracle_after=next(iter(q1))))
    assert rows
    return rows


def evaluate(model, pairs):
    before = predict(model, [p['original'] for p in pairs], len(VOCAB))
    after = predict(model, [p['edited'] for p in pairs], len(VOCAB))
    output = {}
    for kind in sorted(set(p['kind'] for p in pairs)):
        indices = np.array([i for i, p in enumerate(pairs) if p['kind'] == kind])
        pb, pa = before[indices], after[indices]
        q0 = np.zeros_like(pb)
        q1 = np.zeros_like(pa)
        q0[np.arange(len(indices)), [TOKEN2ID[pairs[i]['oracle_before']]
                                    for i in indices]] = 1.
        q1[np.arange(len(indices)), [TOKEN2ID[pairs[i]['oracle_after']]
                                    for i in indices]] = 1.
        native_effect, oracle_effect = pa-pb, q1-q0
        output[kind] = dict(
            n=len(indices), model_tv=float(np.abs(native_effect).sum(-1).mean()/2),
            oracle_tv=float(np.abs(oracle_effect).sum(-1).mean()/2),
            effect_l1_error=float(np.abs(native_effect-oracle_effect).sum(-1).mean()),
            before_answer_probability=float((pb*q0).sum(-1).mean()),
            after_answer_probability=float((pa*q1).sum(-1).mean()))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--nstreams', type=int, default=512)
    p.add_argument('--length', type=int, default=63)
    p.add_argument('--seed', type=int, default=70021)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    pairs = build_pairs(a.nstreams, a.length, a.seed)
    saved = json.loads((a.model_dir/'results.json').read_text())
    results = {}
    for depth in [int(x) for x in a.depths.split(',')]:
        record = next(r for r in saved['records'] if r['depth'] == depth)
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(a.model_dir/f'model_depth{depth}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        results[str(depth)] = evaluate(model, pairs)
        print('depth', depth, {kind: round(z['effect_l1_error'], 4)
                               for kind, z in results[str(depth)].items()}, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(
        config={**vars(a), 'model_dir': str(a.model_dir),
                'output': str(a.output)},
        scores=results, caveat='Grammar-supported interventions preserve exact '
             'prefix support. Score swaps change two digits plus any needed '
             'point/points agreement; person swaps change two names; '
             'question-order swaps change two tokens; wording changes one. '
             'Fresh streams are not a combinatorial train/test holdout.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
