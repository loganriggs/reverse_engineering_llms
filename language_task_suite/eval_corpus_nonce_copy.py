"""Behavioral gate for three-BPE-piece document-local nonce copying."""

from __future__ import annotations

import torch


@torch.no_grad()
def sequence_logprob(model, x, site, candidate):
    """Teacher-force a candidate code after the unchanged source prefix."""
    assert site.shape == candidate.shape and site.shape[1] == 3
    altered = x.clone()
    ix = torch.arange(len(x), device=x.device)
    altered[ix, site[:, 1]] = candidate[:, 0]
    altered[ix, site[:, 2]] = candidate[:, 1]
    logits = model(altered).float()
    positions = logits[ix[:, None], site]
    return positions.log_softmax(-1).gather(
        -1, candidate[:, :, None]).squeeze(-1).sum(-1)


@torch.no_grad()
def greedy_code(model, x, site):
    altered = x.clone()
    ix = torch.arange(len(x), device=x.device)
    predicted = []
    for k in range(3):
        token = model(altered).float()[ix, site[:, k]].argmax(-1)
        predicted.append(token)
        if k < 2:
            altered[ix, site[:, k + 1]] = token
    return torch.stack(predicted, 1)


@torch.no_grad()
def evaluate(model, pair, device, batch_size=8):
    model.eval()
    n = len(pair['left']['x'])
    assert n == len(pair['right']['x'])
    totals = dict(first_piece_both=0, teacher_forced_both=0,
                  greedy_both=0, both_sequence_preference=0,
                  positive_edit=0, piece_top1=0, piece_total=0,
                  greedy_sequence_top1=0, sequence_nll=0.,
                  piece_0=0, piece_1=0, piece_2=0)
    for start in range(0, n, batch_size):
        rows = {arm: {key: value[start:start + batch_size].to(device)
                      for key, value in pair[arm].items()}
                for arm in ('left', 'right')}
        ix = torch.arange(len(rows['left']['x']), device=device)
        own, alternative, margins, greedy_ok = {}, {}, {}, {}
        first_ok, forced_ok = {}, {}
        for arm, other in (('left', 'right'), ('right', 'left')):
            row, opposite = rows[arm], rows[other]
            assert torch.equal(row['site'], opposite['site'])
            assert (row['target'] != opposite['target']).all()
            logits = model(row['x']).float()
            site_logits = logits[ix[:, None], row['site']]
            prediction = site_logits.argmax(-1)
            correct = prediction == row['target']
            first_ok[arm] = correct[:, 0]
            forced_ok[arm] = correct.all(-1)
            totals['piece_top1'] += correct.sum().item()
            totals['piece_total'] += correct.numel()
            for k in range(3):
                totals[f'piece_{k}'] += correct[:, k].sum().item()
            own[arm] = sequence_logprob(model, row['x'], row['site'],
                                        row['target'])
            alternative[arm] = sequence_logprob(
                model, row['x'], row['site'], opposite['target'])
            margins[arm] = own[arm] - alternative[arm]
            totals['sequence_nll'] += (-own[arm]).sum().item()
            greedy = greedy_code(model, row['x'], row['site'])
            greedy_ok[arm] = (greedy == row['target']).all(-1)
            totals['greedy_sequence_top1'] += greedy_ok[arm].sum().item()
        totals['first_piece_both'] += (first_ok['left'] & first_ok['right']).sum().item()
        totals['teacher_forced_both'] += (forced_ok['left'] & forced_ok['right']).sum().item()
        totals['greedy_both'] += (greedy_ok['left'] & greedy_ok['right']).sum().item()
        totals['both_sequence_preference'] += ((margins['left'] > 0) &
                                               (margins['right'] > 0)).sum().item()
        totals['positive_edit'] += (margins['left'] +
                                    margins['right'] > 0).sum().item()
    return {'n': n,
            'first_piece_both': totals['first_piece_both'] / n,
            'teacher_forced_both': totals['teacher_forced_both'] / n,
            'greedy_both': totals['greedy_both'] / n,
            'both_sequence_preference': totals['both_sequence_preference'] / n,
            'positive_edit': totals['positive_edit'] / n,
            'piece_top1': totals['piece_top1'] / totals['piece_total'],
            'piece_top1_by_position': [totals[f'piece_{k}'] / (2 * n)
                                       for k in range(3)],
            'greedy_sequence_top1': totals['greedy_sequence_top1'] / (2 * n),
            'sequence_nll': totals['sequence_nll'] / (2 * n)}


def check_fixture():
    # Altering a candidate's prior generated token must alter x at exactly
    # the next prediction site, not at the first prediction site.
    x = torch.tensor([[9, 8, 7, 6, 5]])
    site = torch.tensor([[1, 2, 3]])
    target = torch.tensor([[1, 2, 3]])
    altered = x.clone()
    altered[0, site[0, 1]] = target[0, 0]
    altered[0, site[0, 2]] = target[0, 1]
    assert altered.tolist() == [[9, 8, 1, 2, 5]]
    print('nonce-copy autoregressive indexing fixture passed')


if __name__ == '__main__':
    check_fixture()
