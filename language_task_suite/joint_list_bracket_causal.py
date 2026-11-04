"""Logit-margin effects of independent list and bracket edits on four-arm panels."""

from __future__ import annotations

import torch


ARMS = ((0, 'round'), (0, 'square'), (1, 'round'), (1, 'square'))
EFFECTS = ('bracket_intended', 'list_intended',
           'bracket_edit_on_list', 'list_edit_on_bracket')


def effects_from_margins(bracket_margin, list_margin):
    """Return intended edit effects and cross-rule spillovers, each n x 2.

    Bracket margin is logit(round token) - logit(square token).
    List margin is logit(shifted numeral) - logit(unshifted numeral).
    Positive intended effects support the known rule. Cross effects should
    ideally vanish because the DGP interventions are independent.
    """
    b = bracket_margin
    l = list_margin
    return {
        'bracket_intended': torch.stack((b[(0, 'round')] - b[(0, 'square')],
                                         b[(1, 'round')] - b[(1, 'square')]), 1),
        'list_intended': torch.stack((l[(1, 'round')] - l[(0, 'round')],
                                      l[(1, 'square')] - l[(0, 'square')]), 1),
        'bracket_edit_on_list': torch.stack((l[(0, 'round')] - l[(0, 'square')],
                                             l[(1, 'round')] - l[(1, 'square')]), 1),
        'list_edit_on_bracket': torch.stack((b[(1, 'round')] - b[(0, 'round')],
                                             b[(1, 'square')] - b[(0, 'square')]), 1),
    }


def _margins(model, batch, device):
    ix = torch.arange(len(batch[(0, 'round')]['x']), device=device)
    bracket_round = batch[(0, 'round')]['target'][:, 0]
    bracket_square = batch[(0, 'square')]['target'][:, 0]
    list_base = batch[(0, 'round')]['target'][:, 1]
    list_shift = batch[(1, 'round')]['target'][:, 1]
    assert (bracket_round != bracket_square).all()
    assert (list_base != list_shift).all()
    bracket, list_ = {}, {}
    for arm in ARMS:
        row = batch[arm]
        shift, style = arm
        assert torch.equal(row['target'][:, 0],
                           bracket_round if style == 'round' else bracket_square)
        assert torch.equal(row['target'][:, 1],
                           list_base if shift == 0 else list_shift)
        out = model(row['x']).float()
        sites = row['site']
        b = out[ix, sites[:, 0]]
        l = out[ix, sites[:, 1]]
        bracket[arm] = b[ix, bracket_round] - b[ix, bracket_square]
        list_[arm] = l[ix, list_shift] - l[ix, list_base]
    return effects_from_margins(bracket, list_)


def _stats(values):
    flat = values.reshape(-1).double()
    return {'mean': float(flat.mean()), 'mean_abs': float(flat.abs().mean()),
            'positive_fraction': float((flat > 0).double().mean()),
            'n_edges': len(flat)}


def _compare(reference, candidate):
    a, b = reference.reshape(-1).double(), candidate.reshape(-1).double()
    centered_a, centered_b = a - a.mean(), b - b.mean()
    denom = centered_a.norm() * centered_b.norm()
    correlation = float((centered_a @ centered_b) / denom) if denom > 0 else None
    return {'mean_abs_error': float((a - b).abs().mean()),
            'sign_agreement': float(((a > 0) == (b > 0)).double().mean()),
            'pearson': correlation}


@torch.no_grad()
def evaluate(teacher, student, panel, device, batch_size=4):
    n = len(panel[(0, 'round')]['x'])
    assert all(len(panel[arm]['x']) == n for arm in ARMS)
    reference = {key: [] for key in EFFECTS}
    candidate = {key: [] for key in reference}
    for start in range(0, n, batch_size):
        batch = {arm: {key: value[start:start + batch_size].to(device)
                       for key, value in panel[arm].items()}
                 for arm in ARMS}
        t = _margins(teacher, batch, device)
        s = _margins(student, batch, device)
        for key in reference:
            reference[key].append(t[key].cpu())
            candidate[key].append(s[key].cpu())
    reference = {key: torch.cat(value) for key, value in reference.items()}
    candidate = {key: torch.cat(value) for key, value in candidate.items()}
    result = {'n_documents': n, 'teacher': {}, 'student': {},
              'student_vs_teacher': {}}
    for key in reference:
        result['teacher'][key] = _stats(reference[key])
        result['student'][key] = _stats(candidate[key])
        result['student_vs_teacher'][key] = _compare(reference[key],
                                                     candidate[key])
    for name, intended, spill in (
            ('bracket', 'bracket_intended', 'bracket_edit_on_list'),
            ('list', 'list_intended', 'list_edit_on_bracket')):
        for which, values in (('teacher', reference), ('student', candidate)):
            top = values[intended].abs().mean().item()
            bottom = values[spill].abs().mean().item()
            result[which][name + '_effect_to_spill_ratio'] = (
                top / bottom if bottom > 0 else None)
    return result


def check():
    # Both rules change their own margins; neither changes the other one.
    b = {(0, 'round'): torch.tensor([2.]),
         (0, 'square'): torch.tensor([-2.]),
         (1, 'round'): torch.tensor([2.]),
         (1, 'square'): torch.tensor([-2.])}
    l = {(0, 'round'): torch.tensor([-3.]),
         (0, 'square'): torch.tensor([-3.]),
         (1, 'round'): torch.tensor([3.]),
         (1, 'square'): torch.tensor([3.])}
    e = effects_from_margins(b, l)
    assert torch.equal(e['bracket_intended'], torch.tensor([[4., 4.]]))
    assert torch.equal(e['list_intended'], torch.tensor([[6., 6.]]))
    assert e['bracket_edit_on_list'].abs().sum() == 0
    assert e['list_edit_on_bracket'].abs().sum() == 0
    print('joint causal-margin fixture passed')


if __name__ == '__main__':
    check()
