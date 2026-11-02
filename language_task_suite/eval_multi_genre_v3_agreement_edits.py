"""Score held-out, grammar-valid agreement edits against the exact oracle."""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_surface_mixture_edits import predict, score
from multi_genre_language import _split_sentences
from multi_genre_v3_agreement import (MultiGenreV3Agreement, NOUN_NUMBER,
                                      NOUN_OTHER_NUMBER)


def build_pairs(gen, n, length):
    rows, labels, _ = gen.sample_batch(n, length+1)
    pairs = []
    for row, labs in zip(rows, labels):
        words = [gen.id2token[i] for i in row]
        if words[0] != 'Agreement':
            continue
        for t in range(1, length+1):
            label = labs[t]
            if label not in ('agreement_main_match',
                             'agreement_main_conflict',
                             'agreement_embedded'):
                continue
            prefix = words[:t]
            history, clause = _split_sentences(prefix[2:])
            start = 2 + sum(map(len, history))
            edits = []
            if label.startswith('agreement_main'):
                changed_subject = prefix.copy()
                changed_subject[start+1] = NOUN_OTHER_NUMBER[clause[1]]
                edits.append(('main_subject_flip', changed_subject))
                changed_attractor = prefix.copy()
                changed_attractor[start+4] = NOUN_OTHER_NUMBER[clause[4]]
                if clause[2] == 'that':
                    changed_attractor[start+5] = ('help' if NOUN_NUMBER[
                        changed_attractor[start+4]] else 'helps')
                    edits.append(('relative_attractor_flip_invariant',
                                  changed_attractor))
                else:
                    edits.append(('preposition_attractor_flip_invariant',
                                  changed_attractor))
                    changed_preposition = prefix.copy()
                    changed_preposition[start+2] = ('beside' if
                        clause[2] == 'near' else 'near')
                    edits.append(('preposition_swap_invariant',
                                  changed_preposition))
            else:
                changed_attractor = prefix.copy()
                changed_attractor[start+4] = NOUN_OTHER_NUMBER[clause[4]]
                edits.append(('embedded_attractor_flip', changed_attractor))
                changed_subject = prefix.copy()
                changed_subject[start+1] = NOUN_OTHER_NUMBER[clause[1]]
                edits.append(('embedded_subject_flip_invariant',
                              changed_subject))
            q0, _ = gen.next_distribution(prefix)
            for kind, edited in edits:
                for j in range(start, t):
                    assert gen.next_distribution(edited[:j])[0].get(
                        edited[j], 0) > 0, (kind, edited, j)
                q1, _ = gen.next_distribution(edited)
                assert (q0 != q1) == (kind in ('main_subject_flip',
                                               'embedded_attractor_flip'))
                pairs.append(dict(kind=kind,
                                  original=[gen.token2id[w] for w in prefix],
                                  edited=[gen.token2id[w] for w in edited],
                                  q_before=q0, q_after=q1))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--seed', type=int, default=9701)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    weights = tuple(float(x) for x in saved['config']['genre_weights'].split(','))
    gen = MultiGenreV3Agreement(seed=a.seed, genre_weights=weights)
    pairs = build_pairs(gen, a.n, a.length)
    assert pairs
    results = {}
    for record in saved['records']:
        depth = record['depth']
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(
            Path(a.model_dir)/f'model_depth{depth}.pt',
            weights_only=True, map_location='cpu'))
        model.eval()
        before = predict(model, [p['original'] for p in pairs], len(gen.vocab))
        after = predict(model, [p['edited'] for p in pairs], len(gen.vocab))
        results[str(depth)] = score(pairs, before, after, gen)
        print('depth', depth, {k: round(v['effect_l1_error'], 3)
                               for k, v in results[str(depth)].items()},
              flush=True)
    Path(a.output).write_text(json.dumps(dict(config=vars(a),
                                             n_pairs=len(pairs),
                                             results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
