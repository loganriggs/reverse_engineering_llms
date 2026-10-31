"""Finite task-margin patch comparison at matched post-c_proj residual dose."""

import argparse
import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_head_holdout import pairs as lexical_pairs
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_format_shift import proposed_pairs as format_pairs
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut, capture_head_states, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


VARIANTS = ('effect__all', 'edit__all', 'svd', 'weight__all')
UNLABELED_VARIANTS = (
    ('raw_pca_all', 'all', 'raw_pca'),
    ('edit_l1_all', 'all', 'edit_l1'),
    ('normalized_pca_all', 'all', 'normalized_pca'),
    ('raw_pca_exclude', 'exclude_target_type', 'raw_pca'),
    ('edit_l1_exclude', 'exclude_target_type', 'edit_l1'),
)
RANDOM_POOL = 64
RANDOM_CONTROLS = 8


def signed_margin(logits, answer_a, answer_b):
    return (logits[:, answer_a] - logits[:, answer_b]).double().numpy()


def run(limit=None, mode='supervised'):
    assert mode in ('supervised', 'unlabeled', 'unlabeled_validation')
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(TOKENIZER)
    config = json.loads((SNAPSHOT / 'config.json').read_text())
    config.pop('step', None)
    model = GPT(GPTConfig(**config)).eval()
    state = torch.load(SNAPSHOT / 'pytorch_model.bin', map_location='cpu',
                       weights_only=True, mmap=True)
    model.load_state_dict(state)
    del state
    model.requires_grad_(False)
    probes = {layer: HeadCut(model.transformer.h[layer].attn)
              for layer in sorted({layer for layer, _ in TARGETS.values()})}
    bases = np.load(ART / 'later_head_rank_pilot_bases.npz')
    rotations = np.load(ART / 'later_head_exact_gauge_rotations.npz')
    selected = json.loads((ART / 'later_head_exact_gauge_finite.json').read_text())['summary']
    if mode.startswith('unlabeled'):
        edit_panel = 'train' if mode == 'unlabeled' else 'validation'
        edit_rotations = np.load(ART / 'later_head_unlabeled_edit_gauge_rotations.npz')
        edit_rows = json.loads((ART / 'later_head_unlabeled_edit_gauge.json').read_text())['records']
        edit_axes = {(r['target'], r['condition'], r['method']): r['selected_axis']
                     for r in edit_rows if r['panel'] == edit_panel
                     and r['selection_rule'] == 'global_energy'}
        variants = tuple(v[0] for v in UNLABELED_VARIANTS)
        variant_lookup = {name: (condition, method)
                          for name, condition, method in UNLABELED_VARIANTS}
    else:
        variants = VARIANTS
    panels = {'lexical_holdout': lexical_pairs(), 'format_shift': format_pairs()}
    records = []
    with torch.no_grad():
        for panel, examples in panels.items():
            for family, base_text, edited_text, answer_a, answer_b in examples:
                if family not in TARGETS:
                    continue
                if limit is not None and sum(r['panel'] == panel and
                                             r['family'] == family for r in records) >= limit:
                    continue
                base_ids = tokenizer.encode(base_text, add_special_tokens=False).ids
                edited_ids = tokenizer.encode(edited_text, add_special_tokens=False).ids
                a = tokenizer.encode(answer_a, add_special_tokens=False).ids
                b = tokenizer.encode(answer_b, add_special_tokens=False).ids
                assert len(a) == len(b) == 1
                assert len(base_ids) == len(edited_ids)
                assert sum(x != y for x, y in zip(base_ids, edited_ids)) == 1
                layer, head = TARGETS[family]
                probe = probes[layer]
                probe.head = head
                base_tensor = torch.tensor(base_ids, dtype=torch.long)[None, :]
                edited_tensor = torch.tensor(edited_ids, dtype=torch.long)[None, :]
                native_before = float(signed_margin(final_logits(model, base_tensor),
                                                    a[0], b[0])[0])
                native_after = float(signed_margin(final_logits(model, edited_tensor),
                                                   a[0], b[0])[0])
                assert native_before > 0 and native_after < 0
                before_state = capture_head_states(model, probes, base_ids)[layer][head]
                after_state = capture_head_states(model, probes, edited_ids)[layer][head]
                raw_delta = (after_state - before_state).numpy().astype(np.float64)
                basis = bases[f'{family}__weight'][:, :4].astype(np.float64)
                probe.basis = torch.tensor(basis.copy(), dtype=torch.float32)
                ids = torch.tensor(base_ids, dtype=torch.long)[None, :]
                baseline = float(signed_margin(final_logits(model, ids), a[0], b[0])[0])
                full_injection = (raw_delta @ basis) @ basis.T
                probe.injection = torch.tensor(full_injection[None], dtype=torch.float32)
                full_effect = float(signed_margin(final_logits(model, ids),
                                                   a[0], b[0])[0] - baseline)
                probe.injection = None

                # Compare changes *after* the learned output projection.
                start = head * probe.head_dim
                output_weight = model.transformer.h[layer].attn.c_proj.weight[
                    :, start:start + probe.head_dim].detach().numpy().astype(np.float64)
                def injection_and_dose(axis):
                    z = float(raw_delta @ axis) * axis
                    return z, float(np.linalg.norm(output_weight @ z))

                null = []
                for index in range(RANDOM_POOL):
                    random_name = f'random_{index}'
                    random_axis = selected[f'{family}__{random_name}']['selected_axis']
                    direction = basis @ rotations[f'{family}__{random_name}'][:, random_axis]
                    direction /= np.linalg.norm(direction)
                    z, dose = injection_and_dose(direction)
                    null.append((index, z, dose))
                # Choose the eight largest-dose random directions using only
                # the head-state edit, not answer labels or final-logit effects.
                controls = sorted(null, key=lambda item: item[2], reverse=True)[
                    :RANDOM_CONTROLS]
                assert controls[-1][2] > 1e-8
                for variant in variants:
                    if mode.startswith('unlabeled'):
                        condition, method = variant_lookup[variant]
                        rotation = edit_rotations[
                            f'{family}__{edit_panel}__{condition}__{method}']
                        axis_index = edit_axes[(family, condition, method)]
                    else:
                        rotation = (np.eye(4) if variant == 'svd'
                                    else rotations[f'{family}__{variant}'])
                        axis_index = selected[f'{family}__{variant}']['selected_axis']
                    axis = basis @ rotation[:, axis_index]
                    axis /= np.linalg.norm(axis)
                    candidate, candidate_dose = injection_and_dose(axis)
                    if candidate_dose <= 1e-8:
                        continue
                    injections = []
                    pair_meta = []
                    for control_index, random_z, random_dose in controls:
                        target = min(candidate_dose, random_dose)
                        injections.extend((candidate * target / candidate_dose,
                                           random_z * target / random_dose))
                        pair_meta.append((control_index, target,
                                          target / candidate_dose,
                                          target / random_dose))
                    injection = np.stack(injections)
                    actual_dose = np.linalg.norm(injection @ output_weight.T, axis=1)
                    expected_dose = np.repeat([m[1] for m in pair_meta], 2)
                    assert np.allclose(actual_dose, expected_dose, rtol=1e-10,
                                       atol=1e-8)
                    probe.injection = torch.tensor(injection, dtype=torch.float32)
                    pred = final_logits(model, ids.repeat(len(injection), 1))
                    effects = signed_margin(pred, a[0], b[0]) - baseline
                    probe.injection = None
                    for j, (control_index, target, cand_fraction, rand_fraction) in enumerate(pair_meta):
                        records.append(dict(panel=panel, family=family,
                            base_prompt=base_text, edited_prompt=edited_text,
                            variant=variant, random_axis=control_index,
                            native_base_margin=native_before,
                            native_edited_margin=native_after,
                            cut_base_margin=baseline,
                            target_post_c_proj_l2=target,
                            candidate_fraction=cand_fraction,
                            random_fraction=rand_fraction,
                            full_rank4_margin_effect=full_effect,
                            candidate_margin_effect=float(effects[2*j]),
                            random_margin_effect=float(effects[2*j+1]),
                            candidate_signed_per_dose=float(
                                effects[2*j] * np.sign(full_effect) / target),
                            random_signed_per_dose=float(
                                effects[2*j+1] * np.sign(full_effect) / target)))
                probe.basis = None
                print(panel, family, base_text, 'full effect', round(full_effect, 3),
                      'records', len(records), flush=True)
    result = dict(description='Frozen rank-four later-head gauges. For each held-out token edit, compare selected axis against eight task-discovery-selected random-rotation axes with the largest local post-c_proj edit dose. Both interventions are scaled to the smaller dose for that pair. Finite full-model correct-minus-foil margin changes are oriented by the full rank-four path sign. The unlabeled mode uses answer-free edit-response fitting and global response-energy axis selection; excluded conditions remove the entire matching edit type. This is a within-reduced-model causal-dose comparison, not a physical necessity test.',
                  mode=mode,
                  random_axis_selection='best mean absolute local task effect on discovery pairs for each random rotation',
                  variants=variants, random_pool=RANDOM_POOL,
                  random_controls=RANDOM_CONTROLS, limit=limit, records=records)
    suffix = '_smoke' if limit is not None else ''
    stem = ('later_head_matched_axis_dose' if mode == 'supervised' else
            'later_head_unlabeled_matched_axis_dose' if mode == 'unlabeled' else
            'later_head_unlabeled_validation_matched_axis_dose')
    path = ART / f'{stem}{suffix}.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('saved', path, len(records), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--mode', choices=('supervised', 'unlabeled',
                                           'unlabeled_validation'),
                        default='supervised')
    args = parser.parse_args()
    run(args.limit, args.mode)
