"""Causal edit panel for two-site verb and object rule mixtures."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_surface_mixture_edits import predict, replacement, score
from eval_surface_mixture_v2_edits import build_pairs as build_verb_pairs
from surface_mixture_v2 import (ADJECTIVES, EVENTS, LEADS, OBJECTS,
                                PLACES, _frame, _indices, _split)
from surface_mixture_v3 import (EVENT_INDEX, OBJECT_INDEX, SurfaceMixtureV3)


def object_edit(gen, prefix, kind):
    history, clause = _split(prefix)
    frame = _frame(clause)
    object_i = OBJECT_INDEX[frame]
    assert len(clause) == object_i
    start = sum(map(len, history))
    edited = list(prefix)
    if kind == 'object_adjective':
        index = start+object_i-1
        old = gen.object_candidates(prefix)['adjective_bigram']
        choices = [a for a in ADJECTIVES if a != prefix[index] and
                   gen.next_distribution(prefix[:index]).get(a, 0) > 0]
        new = next((a for a in choices if gen.object_candidates(
            prefix[:index]+[a]+prefix[index+1:])['adjective_bigram'] != old), None)
        if new is None:
            return None
        edited[index] = new
    elif kind == 'object_cue':
        if frame not in ('lead', 'subject'):
            return None
        index = start+(0 if frame == 'lead' else 2)
        choices = LEADS if frame == 'lead' else ('after', 'before')
        old = gen.object_candidates(prefix)['frame_cue']
        new = next((cue for cue in choices if cue != prefix[index] and
                    gen.object_candidates(prefix[:index]+[cue]+prefix[index+1:])['frame_cue'] != old), None)
        if new is None:
            return None
        edited[index] = new
    elif kind == 'object_induction':
        subject = clause[_indices(frame)[0]]
        prior = [(i, old) for i, old in enumerate(history)
                 if old[_indices(_frame(old))[0]] == subject]
        if not prior:
            return None
        i, old = prior[-1]
        index = sum(map(len, history[:i]))+OBJECT_INDEX[_frame(old)]
        new = replacement(gen, prefix, index, gen.object_lexicon)
        if new is None:
            return None
        edited[index] = new
    elif kind == 'object_event_skip':
        if not history:
            return None
        old = history[-1]
        index = start-len(old)+EVENT_INDEX[_frame(old)]
        objects = gen.object_lexicon
        target = lambda event: objects[(EVENTS.index(event)*13+5)%len(objects)]
        new = replacement(gen, prefix, index, EVENTS, distinct=target)
        if new is None:
            return None
        edited[index] = new
    elif kind == 'object_nuisance_place':
        if not history:
            return None
        old = history[-1]
        index = start-len(old)+_indices(_frame(old))[2]
        new = replacement(gen, prefix, index, PLACES)
        if new is None:
            return None
        edited[index] = new
    else:
        raise ValueError(kind)
    assert edited != prefix
    q0, q1 = gen.next_distribution(prefix), gen.next_distribution(edited)
    assert (q0 == q1) == (kind == 'object_nuisance_place')
    return edited, q0, q1


def build_object_pairs(gen, n, length):
    ids, labels, _ = gen.sample_batch(n, length+1)
    pairs = []
    for row, labs in zip(ids, labels):
        tokens = [gen.id2token[i] for i in row]
        for t in range(1, length+1):
            if labs[t] != 'mixed_object':
                continue
            prefix = tokens[:t]
            frame = _frame(_split(prefix)[1])
            for kind in ('object_adjective', 'object_cue', 'object_induction',
                         'object_event_skip', 'object_nuisance_place'):
                result = object_edit(gen, prefix, kind)
                if result is None:
                    continue
                edited, q0, q1 = result
                pairs.append(dict(kind=kind, frame=frame,
                                  original=[gen.token2id[x] for x in prefix],
                                  edited=[gen.token2id[x] for x in edited],
                                  q_before=q0, q_after=q1))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=8701)
    p.add_argument('--object-vocab-size', type=int, default=None)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    gen = SurfaceMixtureV3(seed=a.seed,
                           object_vocab_size=a.object_vocab_size)
    objects = build_object_pairs(gen, a.n, a.length)
    verbs = build_verb_pairs(SurfaceMixtureV3(
        seed=a.seed, object_vocab_size=a.object_vocab_size), a.n, a.length)
    pairs = verbs+objects
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    results = {}
    for directory in a.model_dirs:
        folder = Path(directory)
        saved = json.loads((folder/'results.json').read_text())
        for record in saved['records']:
            depth = record['depth']
            model = BilinearGPT(Config(**record['model_config']))
            model.load_state_dict(torch.load(folder/f'model_depth{depth}.pt',
                                             map_location='cpu', weights_only=True))
            model.eval()
            before = predict(model, [pair['original'] for pair in pairs], len(gen.vocab))
            after = predict(model, [pair['edited'] for pair in pairs], len(gen.vocab))
            name = f'{folder.name}_depth{depth}'
            results[name] = score(pairs, before, after, gen)
            print(name, {k: round(v['effect_l1_error'], 3)
                         for k, v in results[name].items()}, flush=True)
    output = dict(config=vars(a), pair_counts=dict(Counter(p['kind'] for p in pairs)),
                  results=results,
                  caveat='All edits change one supported visible token and are '
                         'scored against exact object/verb mixture changes. '
                         'An edit can change the probability of its own prefix; '
                         'this test compares matched next-token effects.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
