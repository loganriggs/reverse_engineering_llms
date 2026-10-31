"""Agreement-head reduction x answer-free group test with token-swap exclusion."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_bracket_group_competent_transfer import margin, swap
from flagship_explicit_task_head_holdout import pairs as lexical_pairs
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_format_shift import proposed_pairs as format_pairs
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


MEASURES = ('weight', 'corpus', 'task_delta', 'mixed', 'random')
TARGET = 'subject_verb_agreement'
LAYER, HEAD = 11, 3


def head_state(model, probe, ids):
    final_logits(model, torch.tensor(ids, dtype=torch.long)[None, :], stop_at=LAYER)
    start = HEAD * probe.head_dim
    return probe.last_input[0, -1, start:start + probe.head_dim].clone()


def run():
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(TOKENIZER)
    edit_meta = json.loads((ART / 'later_head_unlabeled_edits.json').read_text())
    fit = [x for x in edit_meta['panels']['train'] if x['type'] == 'noun_number']
    assert len(fit) == 80
    tests = []
    for source, panel in ((lexical_pairs(), 'lexical'), (format_pairs(), 'format')):
        for family, base, edited, answer_a, answer_b in source:
            if family != TARGET:
                continue
            before = tokenizer.encode(base, add_special_tokens=False).ids
            after = tokenizer.encode(edited, add_special_tokens=False).ids
            a = tokenizer.encode(answer_a, add_special_tokens=False).ids
            b = tokenizer.encode(answer_b, add_special_tokens=False).ids
            assert len(a) == len(b) == 1
            tests.append(dict(panel=panel, base_text=base, edited_text=edited,
                              base_ids=before, edited_ids=after, answer_a=a[0], answer_b=b[0],
                              token_swap=swap(before, after)))
    assert len(tests) == 20
    held_swaps = {x['token_swap'] for x in tests}
    strict_fit = [x for x in fit if swap(x['base_ids'], x['edited_ids']) not in held_swaps]
    assert len(strict_fit) == 44
    assert not {swap(x['base_ids'], x['edited_ids']) for x in strict_fit} & held_swaps
    with np.load(ART / 'later_head_rank_pilot_bases.npz') as saved:
        bases = {name: torch.tensor(saved[f'{TARGET}__{name}'][:, :4].copy(),
                                    dtype=torch.float32) for name in MEASURES}
        assert all(np.linalg.norm(saved[f'{TARGET}__{name}'][:, :4].T @
                                  saved[f'{TARGET}__{name}'][:, :4] - np.eye(4)) < 1e-12
                   for name in MEASURES)
    config = json.loads((SNAPSHOT / 'config.json').read_text())
    config.pop('step', None)
    model = GPT(GPTConfig(**config)).eval()
    weights = torch.load(SNAPSHOT / 'pytorch_model.bin', map_location='cpu',
                         weights_only=True, mmap=True)
    model.load_state_dict(weights)
    del weights
    model.requires_grad_(False)
    probe = HeadCut(model.transformer.h[LAYER].attn)
    probe.head = HEAD

    with torch.no_grad():
        fit_vectors = {panel: np.stack([(head_state(model, probe, x['edited_ids']) -
                                         head_state(model, probe, x['base_ids'])).numpy()
                                        for x in subset]).astype(np.float64)
                       for panel, subset in (('all80', fit), ('identity_holdout44', strict_fit))}
    rotations = {}
    spectra = {}
    for panel, vectors in fit_vectors.items():
        for name, basis in bases.items():
            coordinates = vectors @ basis.numpy().astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(coordinates.T @ coordinates / len(coordinates))
            key = f'{panel}__{name}'
            rotations[key] = torch.tensor(eigenvectors[:, ::-1].copy(), dtype=torch.float32)
            spectra[key] = eigenvalues[::-1].tolist()
    rows = []
    with torch.no_grad():
        for index, case in enumerate(tests):
            before, after = case['base_ids'], case['edited_ids']
            a, b = case['answer_a'], case['answer_b']
            delta = (head_state(model, probe, after) - head_state(model, probe, before))[None, :]
            native_before = margin(model, before, a, b)
            native_after = margin(model, after, a, b)
            assert native_before > 0 and native_after < 0
            probe.basis = torch.eye(probe.head_dim)
            probe.injection = delta
            full_head = margin(model, before, a, b) - native_before
            probe.injection = None
            effects, task_margins = {}, {}
            for name, basis in bases.items():
                probe.basis = basis
                baseline = margin(model, before, a, b)
                edited_margin = margin(model, after, a, b)
                task_margins[name] = dict(base=baseline, edited=edited_margin)
                effects[name] = {}
                for panel in fit_vectors:
                    ranks = {}
                    for rank in (1, 2, 4):
                        subspace = basis @ rotations[f'{panel}__{name}'][:, :rank] if rank < 4 else basis
                        probe.injection = (delta @ subspace) @ subspace.T
                        ranks[f'rank{rank}'] = margin(model, before, a, b) - baseline
                        probe.injection = None
                    effects[name][panel] = ranks
                assert abs(effects[name]['all80']['rank4'] -
                           effects[name]['identity_holdout44']['rank4']) < 1e-5
            probe.basis = None
            rows.append(dict(panel=case['panel'], base_text=case['base_text'],
                             edited_text=case['edited_text'], token_swap=case['token_swap'],
                             native_head_path=full_head, effects=effects,
                             rank4_task_margins=task_margins))
            if (index + 1) % 5 == 0:
                print('validation', index + 1, '/', len(tests), flush=True)
    summary = []
    for group_name in ('lexical', 'format', 'all'):
        group = rows if group_name == 'all' else [x for x in rows if x['panel'] == group_name]
        norm = np.mean([abs(x['native_head_path']) for x in group])
        for name in MEASURES:
            for fit_panel in fit_vectors:
                part = [x['effects'][name][fit_panel] for x in group]
                summary.append(dict(test_panel=group_name, measure=name, fit_panel=fit_panel,
                    n=len(group), mean_abs_native_head_path=float(norm),
                    rank4_both_forced_choice_correct=sum(
                        x['rank4_task_margins'][name]['base'] > 0 and
                        x['rank4_task_margins'][name]['edited'] < 0 for x in group),
                    relative_mae_vs_native={f'rank{rank}': float(np.mean([
                        abs(e[f'rank{rank}'] - x['native_head_path'])
                        for e, x in zip(part, group)]) / norm) for rank in (1, 2, 4)},
                    relative_mae_vs_own_rank4={f'rank{rank}': float(np.mean([
                        abs(e[f'rank{rank}'] - e['rank4']) for e in part]) /
                        max(np.mean([abs(e['rank4']) for e in part]), 1e-20)) for rank in (1, 2)}))
    result = dict(description='Agreement head factorial: five frozen rank-four cuts x answer-free edit covariance fit; 20 competent independent contexts, including a fit withholding all validation token-swap identities',
        target=TARGET, layer=LAYER, head=HEAD, fit_counts={k: len(v) for k, v in fit_vectors.items()},
        test_count=len(tests), test_swap_count=len(held_swaps),
        strict_fit_excludes_all_test_swaps=True, task_labels_used_for_group_fit=False,
        fit_spectra=spectra, rows=rows, summary=summary)
    (ART / 'agreement_reduction_group_factorial.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(ART / 'agreement_reduction_group_factorial_rotations.npz',
                        **{k: v.numpy() for k, v in rotations.items()})
    for row in summary:
        if row['test_panel'] == 'all':
            print(row, flush=True)


if __name__ == '__main__':
    run()
