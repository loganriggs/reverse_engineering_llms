"""Grammar-valid causal rule edits for the six-frame joint-rule language."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

from eval_surface_bond_gauges import bases_for
from eval_surface_mixture_edits import predict, replacement, score
from surface_mixture_v2 import (ADJECTIVES, LEADS, PEOPLE, PLACES,
                                VERBS, SurfaceMixtureV2, _frame, _indices,
                                _metadata, _split)
from train_surface_mixture import make_batch


ADJECTIVE_INDEX = {'lead': 5, 'subject': 8, 'at': 7,
                   'when': 8, 'in_place': 6, 'during': 6}


def edited_pair(gen, prefix, kind):
    history, clause = _split(prefix)
    frame = _frame(clause)
    subject_i, verb_i, _, _ = _indices(frame)
    assert len(clause) == verb_i
    start = sum(map(len, history))
    edited = list(prefix)
    if kind == 'cue':
        if frame not in ('lead', 'subject'):
            return None
        index = start + (0 if frame == 'lead' else 2)
        choices = LEADS if frame == 'lead' else ('after', 'before')
        old = gen.candidates(prefix)['lead']
        for option in choices:
            if option == prefix[index]:
                continue
            trial = list(prefix)
            trial[index] = option
            if gen.candidates(trial)['lead'] != old:
                edited = trial
                break
        if edited == prefix:
            return None
    elif kind == 'lexical':
        previous_subjects = {_metadata(old)['subject'] for old in history}
        if clause[subject_i] in previous_subjects:
            return None
        index = start + subject_i
        old = gen.candidates(prefix)['lexical']
        choices = [person for person in PEOPLE
                   if person != clause[subject_i] and person not in previous_subjects
                   and gen.next_distribution(prefix[:index]).get(person, 0) > 0]
        new = next((person for person in choices
                    if gen.candidates(prefix[:index]+[person]+prefix[index+1:])['lexical'] != old),
                   None)
        if new is None:
            return None
        edited[index] = new
    elif kind == 'induction':
        prior = [(i, old) for i, old in enumerate(history)
                 if _metadata(old)['subject'] == clause[subject_i]]
        if not prior:
            return None
        i, old = prior[-1]
        index = sum(map(len, history[:i])) + _indices(_frame(old))[1]
        new = replacement(gen, prefix, index, VERBS)
        if new is None:
            return None
        edited[index] = new
    elif kind == 'place_skip':
        if not history:
            return None
        old = history[-1]
        index = start-len(old)+_indices(_frame(old))[2]
        target = lambda place: VERBS[(PLACES.index(place)*7+2)%len(VERBS)]
        new = replacement(gen, prefix, index, PLACES, distinct=target)
        if new is None:
            return None
        edited[index] = new
    elif kind == 'nuisance_adjective':
        if not history:
            return None
        old = history[-1]
        index = start-len(old)+ADJECTIVE_INDEX[_frame(old)]
        new = replacement(gen, prefix, index, ADJECTIVES)
        if new is None:
            return None
        edited[index] = new
    else:
        raise ValueError(kind)
    assert edited != prefix
    q0, q1 = gen.next_distribution(prefix), gen.next_distribution(edited)
    assert (q0 == q1) == (kind == 'nuisance_adjective')
    return edited, q0, q1


def build_pairs(gen, n, length):
    ids, labels, _ = gen.sample_batch(n, length+1)
    pairs = []
    for row, labs in zip(ids, labels):
        tokens = [gen.id2token[i] for i in row]
        for t in range(1, length+1):
            if labs[t] != 'mixed_verb':
                continue
            prefix = tokens[:t]
            frame = _frame(_split(prefix)[1])
            for kind in ('cue', 'lexical', 'induction', 'place_skip',
                         'nuisance_adjective'):
                result = edited_pair(gen, prefix, kind)
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
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_v2_models')
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=8701)
    p.add_argument('--subject-repeat-mass', type=float, default=None,
                   help='Override the checkpoint DGP repeat mass')
    p.add_argument('--truncate-layer', type=int, default=None)
    p.add_argument('--basis', choices=('physical', 'random', 'activation_pca',
                                       'weight_svd'), default='weight_svd')
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    repeat_mass = (a.subject_repeat_mass if a.subject_repeat_mass is not None
                   else saved.get('generator_kwargs', {}).get(
                       'subject_repeat_mass', .4))
    gen = SurfaceMixtureV2(seed=a.seed,
                           subject_repeat_mass=repeat_mass)
    pairs = build_pairs(gen, a.n, a.length)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    results = {}
    results_by_frame = {}
    originals = [pair['original'] for pair in pairs]
    edits = [pair['edited'] for pair in pairs]
    for record in saved['records']:
        depth = record['depth']
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{depth}.pt',
                                         weights_only=True, map_location='cpu'))
        model.eval()
        hook = None
        if a.truncate_layer is not None:
            assert 0 <= a.truncate_layer < depth
            bond = model.blocks[a.truncate_layer].attn.out
            captured = []
            def collect(_, inputs):
                captured.append(inputs[0].detach())
            hook = bond.register_forward_pre_hook(collect)
            xfit, _ = make_batch(SurfaceMixtureV2(
                seed=8501, subject_repeat_mass=repeat_mass), 256,
                                 a.length, 'cpu')
            with torch.no_grad():
                for batch in xfit.split(32):
                    model(batch)
            hook.remove()
            basis = bases_for(torch.cat(captured), bond.weight.detach())[a.basis]
            uk = basis[:, :a.width]
            def truncate(_, inputs):
                x = inputs[0]
                return ((x@uk)@uk.T,)
            hook = bond.register_forward_pre_hook(truncate)
        before = predict(model, originals, len(gen.vocab))
        after = predict(model, edits, len(gen.vocab))
        if hook is not None:
            hook.remove()
        results[str(depth)] = score(pairs, before, after, gen)
        results_by_frame[str(depth)] = {}
        for frame in ('lead', 'subject', 'at', 'when', 'in_place', 'during'):
            indices = [i for i, pair in enumerate(pairs) if pair['frame'] == frame]
            results_by_frame[str(depth)][frame] = score(
                [pairs[i] for i in indices], before[indices], after[indices], gen)
        print('depth', depth, {k: round(v['effect_l1_error'], 3)
                               for k, v in results[str(depth)].items()}, flush=True)
    by_frame = defaultdict(lambda: defaultdict(int))
    for pair in pairs:
        by_frame[pair['frame']][pair['kind']] += 1
    output = dict(config={**vars(a), 'effective_subject_repeat_mass': repeat_mass},
                  n_pairs=len(pairs),
                  pairs_by_frame={k: dict(v) for k, v in by_frame.items()},
                  results=results, results_by_frame=results_by_frame,
                  caveat='One-token edits are valid under the visible-prefix grammar. '
                         'Some edits alter the probability of their edited prefix; '
                         'effects compare next-verb predictions on matched prefixes.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
