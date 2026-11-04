"""Compare rank-four per-head OV reductions on ordinary-order rule panels."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_corpus_rule_reference_transfer import KINDS, one_kind
from fit_corpus_list_bracket_ov import training_rows
from ov_linear_reduction import activation_grams, attentions, replace_ov
from train_corpus_delimiter_bilinear import _model


@torch.no_grad()
def fidelity(teacher, student, records, device, batch_size=4):
    errors, effects_t, effects_s = [], [], []
    site_kl_sum = background_kl_sum = 0.
    site_count = background_count = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        n = len(batch)
        ix = torch.arange(n, device=device)
        xn = torch.stack([r['x_natural'] for r in batch]).to(device)
        xe = torch.stack([r['x_edit'] for r in batch]).to(device)
        yn = torch.stack([r['y_natural'] for r in batch]).to(device)
        pn = torch.tensor([r['pos_natural'] for r in batch], device=device)
        pe = torch.tensor([r['pos_edit'] for r in batch], device=device)
        tn = torch.tensor([r['token_natural'] for r in batch], device=device)
        te = torch.tensor([r['token_edit'] for r in batch], device=device)
        x = torch.cat((xn, xe))
        t = teacher(x).float()
        s = student(x).float()
        t_n, t_e = t[:n][ix, pn], t[n:][ix, pe]
        s_n, s_e = s[:n][ix, pn], s[n:][ix, pe]
        margin = lambda a, b: a[ix, tn] - a[ix, te] - (b[ix, tn] - b[ix, te])
        effect_t, effect_s = margin(t_n, t_e), margin(s_n, s_e)
        effects_t.append(effect_t.cpu())
        effects_s.append(effect_s.cpu())
        errors.append((effect_s - effect_t).abs().cpu())
        for reference, candidate in ((t_n, s_n), (t_e, s_e)):
            site_kl_sum += F.kl_div(candidate.log_softmax(-1),
                                     reference.softmax(-1),
                                     reduction='sum').item()
            site_count += n
        positions = torch.arange(0, xn.shape[1], 32, device=device)
        valid = ((yn[:, positions] != -100) &
                 (positions[None, :] != pn[:, None]))
        if valid.any():
            reference, candidate = t[:n, positions][valid], s[:n, positions][valid]
            background_kl_sum += F.kl_div(candidate.log_softmax(-1),
                                          reference.softmax(-1),
                                          reduction='sum').item()
            background_count += int(valid.sum())
    a, b = torch.cat(effects_t).double(), torch.cat(effects_s).double()
    centered_a, centered_b = a-a.mean(), b-b.mean()
    denom = centered_a.norm()*centered_b.norm()
    return {'n': len(records),
            'effect_mean_abs_error': torch.cat(errors).double().mean().item(),
            'effect_sign_agreement': ((a > 0) == (b > 0)).double().mean().item(),
            'effect_pearson': ((centered_a @ centered_b)/denom).item()
            if denom > 0 else None,
            'site_teacher_kl': site_kl_sum/site_count,
            'sampled_background_teacher_kl': background_kl_sum/background_count,
            'sampled_background_positions': background_count}


@torch.no_grad()
def keep_one_full_head(model):
    """Same retained OV channel count as rank4 in four heads per layer."""
    selected = []
    for layer_index, layer in enumerate(attentions(model)):
        norms = []
        for head in range(layer.n_head):
            sl = slice(head*layer.d_head, (head+1)*layer.d_head)
            product = layer.o.weight[:, sl] @ layer.v.weight[sl]
            norms.append(torch.linalg.matrix_norm(product).item())
        best = max(range(layer.n_head), key=lambda h: norms[h])
        selected.append({'layer': layer_index, 'head': best,
                         'product_norms': norms})
        for head in range(layer.n_head):
            if head != best:
                sl = slice(head*layer.d_head, (head+1)*layer.d_head)
                layer.o.weight[:, sl].zero_()
                layer.v.weight[sl].zero_()
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--panels', type=Path, nargs='+', required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--fit-rows-per-arm', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    if a.rank != 4:
        raise ValueError('The one-full-head matched-channel control requires rank 4')
    torch.set_num_threads(4)
    list_data = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    bracket_data = torch.load(a.bracket_cache, map_location='cpu', weights_only=True)
    list_rows, bracket_rows = training_rows(list_data, bracket_data,
                                           a.fit_rows_per_arm)
    gram_rows = {key: torch.cat((list_rows[key], bracket_rows[key]))
                 for key in ('x', 'y')}
    panels = [torch.load(path, map_location='cpu', weights_only=True)
              for path in a.panels]
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    report = {'config': {key: [str(x) for x in value] if isinstance(value, list)
                         else str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'scope': 'Rank-four per-head OV product approximations versus '
                       'one full head per layer, matched at 16 retained OV '
                       'channels per layer (4 of 16 in each of four heads '
                       'versus one of four complete 16-channel heads). '
                       'All QK, MLP, embedding, unembedding unchanged. '
                       'Data metric estimated on separate-rule training '
                       'windows, never on test panels. Caches with token '
                       'IDs remain local. This tests functional fidelity, '
                       'not interpretability or circuit recovery.',
              'models': {}}
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        teacher = _model(50257, 256, 2, 64, 4, a.device, 'none')
        teacher.load_state_dict(saved['state_dict'])
        teacher.eval()
        grams, counts = activation_grams(teacher, gram_rows, a.device)
        model_result = {'gram_counts': counts, 'methods': {}}
        report['models'][path.stem] = model_result
        for method in ('weight', 'data', 'head_top1'):
            student = deepcopy(teacher).eval()
            reduction_errors = (replace_ov(student, grams, a.rank, method)
                                if method != 'head_top1'
                                else keep_one_full_head(student))
            method_result = {'reduction_errors': reduction_errors, 'panels': {}}
            model_result['methods'][method] = method_result
            for panel_path, panel in zip(a.panels, panels):
                rows = panel['records']
                by_kind = {}
                for kind in KINDS:
                    chosen = [row for row in rows if row['kind'] == kind]
                    with torch.no_grad():
                        behavior = one_kind(student, chosen, a.device,
                                            a.batch_size, tokenizer)
                    by_kind[kind] = {
                        'behavior': behavior,
                        'teacher_fidelity': fidelity(
                            teacher, student, chosen, a.device, a.batch_size)}
                    print(path.stem, method, panel_path.stem, kind,
                          'paired top1', round(behavior['paired_top1_fraction'], 3),
                          'effect MAE', round(by_kind[kind]['teacher_fidelity']
                                              ['effect_mean_abs_error'], 3),
                          flush=True)
                method_result['panels'][panel_path.stem] = by_kind
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, indent=2) + '\n')
            del student
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
