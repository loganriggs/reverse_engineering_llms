"""Patch mixture-tag contrasts through one attention-output bond.

Fit bases on separate prefixes, optionally excluding all rule conflicts.
Evaluate whether patching donor coordinates into the recipient model moves
bigram-vs-induction log odds toward the donor behavior on held-out prefixes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_mixed_rule_weight_sweep import MIXTURES
from mixed_rule_oracle import observable_distributions_for_sequence


def positions_for(gen, ids):
    result = []
    conflict = np.zeros((len(ids), len(ids[0])), dtype=bool)
    for b, row in enumerate(ids):
        tokens = [gen.id2token[i] for i in row]
        for t in range(1, len(row)):
            context = tokens[:t]
            proposals = []
            for cls, fn in [('bigram', gen._try_bigram),
                            ('trigram', gen._try_trigram),
                            ('skip_bigram', gen._try_skip_bigram),
                            ('induction', gen._try_induction)]:
                token, _ = fn(context)
                if token is not None:
                    proposals.append((cls, token))
            conflict[b, t] = len({token for _, token in proposals}) > 1
            if len(proposals) == 2 and {cls for cls, _ in proposals} == {
                    'bigram', 'induction'} and conflict[b, t]:
                bigram = next(token for cls, token in proposals if cls == 'bigram')
                induction = next(token for cls, token in proposals if cls == 'induction')
                result.append((b, t, gen.token2id[bigram], gen.token2id[induction]))
    return result, conflict


def tagged_x(ids, tag):
    z = torch.tensor(ids, dtype=torch.long)
    prefix = torch.full((len(ids), 1), tag, dtype=torch.long)
    return torch.cat([prefix, z[:, :-1]], 1)


@torch.no_grad()
def forward_collect(model, bond, x):
    captured = []
    def capture(_, inputs):
        captured.append(inputs[0].detach())
    hook = bond.register_forward_pre_hook(capture)
    logits = torch.cat([model(batch)[0] for batch in x.split(32)])
    hook.remove()
    return logits.float(), torch.cat(captured)


@torch.no_grad()
def forward_patch(model, bond, x, donor_acts, u, width, positions):
    outputs = []
    for batch, donor in zip(x.split(32), donor_acts.split(32)):
        uk = u[:, :width]
        def patch(_, inputs):
            original = inputs[0]
            changed = original + ((donor-original) @ uk) @ uk.T
            if positions == 'tag_only':
                changed[:, 1:] = original[:, 1:]
            elif positions == 'non_tag_only':
                changed[:, :1] = original[:, :1]
            return (changed,)
        hook = bond.register_forward_pre_hook(patch)
        outputs.append(model(batch)[0].float())
        hook.remove()
    return torch.cat(outputs)


def make_bases(acts_recipient, acts_donor, fit_mask, weight, n_head):
    d = acts_recipient.shape[-1]
    delta = (acts_donor-acts_recipient)[fit_mask].reshape(-1, d).double()
    activations = acts_recipient[fit_mask].reshape(-1, d).double()
    _, u_delta = torch.linalg.eigh(delta.T @ delta)
    _, u_activation = torch.linalg.eigh(activations.T @ activations)
    _, _, vh = torch.linalg.svd(weight.double())
    rng = np.random.default_rng(7831)
    random, _ = np.linalg.qr(rng.normal(size=(d, d)))
    bases = {'physical': torch.eye(d),
            'random': torch.tensor(random, dtype=torch.float32),
            'activation_pca': torch.flip(u_activation, (1,)).float(),
            'weight_svd': vh.T.float(),
            'tag_contrast_pca': torch.flip(u_delta, (1,)).float()}
    head_dim = d//n_head
    for h in range(n_head):
        selected = list(range(h*head_dim, (h+1)*head_dim))
        order = selected + [i for i in range(d) if i not in selected]
        bases[f'head{h}_first'] = torch.eye(d)[:, order]
    return bases


@torch.no_grad()
def weight_only_tag_contrast(model, length, ordinary_vocab=26,
                             recipient_tag=26, donor_tag=28):
    """First-bond tag contrast for every token/position, computed from weights.

    At positions after the tag, all non-tag keys and values are identical in
    the two runs. Only the score-weighted value of the tag can differ. A
    uniform token-by-position SVD therefore needs no sampled language text.
    """
    from model import apply_rotary_emb

    block = model.blocks[0]
    attn = block.attn
    h, d = attn.n_head, attn.head_dim
    tokens = model.embed.weight[:ordinary_vocab]
    tags = model.embed.weight[[recipient_tag, donor_tag]]
    if block.use_rmsnorm:
        tokens = block.norm_attn(tokens)
        tags = block.norm_attn(tags)
    cos, sin = attn.rotary(torch.empty(1, length, h, d))

    def score_for(tag):
        query_input = tokens[:, None, :].expand(-1, length, -1)
        q1 = attn.q1(query_input).reshape(ordinary_vocab, length, h, d)
        q2 = attn.q2(query_input).reshape(ordinary_vocab, length, h, d)
        k1 = attn.k1(tag).reshape(1, 1, h, d)
        k2 = attn.k2(tag).reshape(1, 1, h, d)
        q1 = apply_rotary_emb(q1, cos, sin)
        q2 = apply_rotary_emb(q2, cos, sin)
        k1 = apply_rotary_emb(k1, cos[:, :1], sin[:, :1])
        k2 = apply_rotary_emb(k2, cos[:, :1], sin[:, :1])
        if attn.qk_norm:
            q1, q2 = F.normalize(q1, dim=-1), F.normalize(q2, dim=-1)
            k1, k2 = F.normalize(k1, dim=-1), F.normalize(k2, dim=-1)
        return ((q1*k1).sum(-1)/d)*((q2*k2).sum(-1)/d)

    values = attn.v(tags).reshape(2, h, d)
    scores_recipient = score_for(tags[0])
    scores_donor = score_for(tags[1])
    delta = (scores_donor[..., None]*values[1]
             - scores_recipient[..., None]*values[0])[:, 1:]
    delta = delta.reshape(-1, model.config.n_embd).double()
    _, singular, vh = torch.linalg.svd(delta, full_matrices=True)
    basis = vh.T.float()
    assert float(singular[2*h:].square().sum()/singular.square().sum()) < 1e-8
    return basis, delta


def odds(logits, positions):
    logp = F.log_softmax(logits, -1)
    return torch.stack([logp[b, t, bigram]-logp[b, t, induction]
                        for b, t, bigram, induction in positions])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/conditional_mixed_rules_v1')
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--fit-contexts', choices=['all', 'no_conflict'], default='no_conflict')
    p.add_argument('--fit-exclude-tag', action='store_true')
    p.add_argument('--patch-positions', choices=['all', 'tag_only', 'non_tag_only'],
                   default='all')
    p.add_argument('--nfit', type=int, default=128)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--fit-seed', type=int, default=7801)
    p.add_argument('--test-seed', type=int, default=7802)
    p.add_argument('--donor-mixture', choices=['bigram_heavy', 'induction_heavy'],
                   default='induction_heavy')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from generator import LanguageGenerator, ALL_RULES
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    model_dir = Path(a.model_dir)
    config = Config(**json.loads((model_dir/'model_config.json').read_text()))
    assert a.length == config.seq_len
    model = BilinearGPT(config)
    model.load_state_dict(torch.load(model_dir/'model.pt', weights_only=True,
                                     map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    fit_gen = LanguageGenerator(ALL_RULES, mixing_weights=MIXTURES['baseline'],
                                mode='mixed', seed=a.fit_seed)
    test_gen = LanguageGenerator(ALL_RULES, mixing_weights=MIXTURES['baseline'],
                                 mode='mixed', seed=a.test_seed)
    donor_gen = LanguageGenerator(ALL_RULES,
                                  mixing_weights=MIXTURES[a.donor_mixture],
                                  mode='mixed', seed=7803)
    donor_tag = 27 if a.donor_mixture == 'bigram_heavy' else 28
    fit_ids, _ = fit_gen.sample_batch(a.nfit, a.length)
    test_ids, _ = test_gen.sample_batch(a.ntest, a.length)
    fit_positions, fit_conflict = positions_for(fit_gen, fit_ids)
    test_positions, test_conflict = positions_for(test_gen, test_ids)
    x_fit_recipient = tagged_x(fit_ids, 26)
    x_fit_donor = tagged_x(fit_ids, donor_tag)
    _, fit_acts_recipient = forward_collect(model, bond, x_fit_recipient)
    _, fit_acts_donor = forward_collect(model, bond, x_fit_donor)
    fit_mask = torch.ones((a.nfit, a.length), dtype=torch.bool)
    if a.fit_contexts == 'no_conflict':
        fit_mask = ~torch.from_numpy(fit_conflict)
    if a.fit_exclude_tag:
        fit_mask[:, 0] = False
    fit_delta = (fit_acts_donor-fit_acts_recipient)[fit_mask].reshape(
        -1, config.n_embd).double()
    singular_values = torch.linalg.svdvals(fit_delta)
    variance = singular_values.square()
    if a.layer == 0 and a.fit_exclude_tag:
        theoretical_rank = 2*config.n_head
        assert float(variance[theoretical_rank:].sum()/variance.sum()) < 1e-8
    contrast_variance_captured = {
        str(k): float(variance[:k].sum()/variance.sum())
        for k in (1, 2, 4, 8, 16, 32) if k <= config.n_embd}
    head_dim = config.n_embd//config.n_head
    head_contrast_energy = {
        str(h): float(fit_delta[:, h*head_dim:(h+1)*head_dim].square().sum()
                      /fit_delta.square().sum())
        for h in range(config.n_head)}
    bases = make_bases(fit_acts_recipient, fit_acts_donor, fit_mask,
                       bond.weight.detach(), config.n_head)
    weight_only_formula_error = None
    if a.layer == 0:
        basis, predicted_delta = weight_only_tag_contrast(
            model, a.length, donor_tag=donor_tag)
        bases['weight_only_tag_contrast'] = basis
        token_grid = torch.arange(26)[:, None].expand(-1, a.length-1)
        probe_recipient = torch.cat((torch.full((26, 1), 26), token_grid), 1)
        probe_donor = torch.cat((torch.full((26, 1), donor_tag), token_grid), 1)
        _, probe_recipient_acts = forward_collect(model, bond, probe_recipient)
        _, probe_donor_acts = forward_collect(model, bond, probe_donor)
        observed_delta = (probe_donor_acts-probe_recipient_acts)[:, 1:].reshape(
            -1, config.n_embd).double()
        weight_only_formula_error = float((predicted_delta-observed_delta)
                                          .square().mean().sqrt())
        assert weight_only_formula_error < 1e-5, weight_only_formula_error
    x_recipient = tagged_x(test_ids, 26)
    x_donor = tagged_x(test_ids, donor_tag)
    recipient_logits, _ = forward_collect(model, bond, x_recipient)
    donor_logits, donor_acts = forward_collect(model, bond, x_donor)
    recipient_logp = F.log_softmax(recipient_logits, -1)
    donor_logp = F.log_softmax(donor_logits, -1)
    recipient_p = recipient_logp.exp()
    null_mask = torch.zeros((a.ntest, a.length), dtype=torch.bool)
    oracle_recipient = []
    oracle_donor = []
    for b, ids in enumerate(test_ids):
        tokens = [test_gen.id2token[i] for i in ids]
        q_rec = observable_distributions_for_sequence(test_gen, tokens)
        q_don = observable_distributions_for_sequence(donor_gen, tokens)
        oracle_recipient.append(q_rec)
        oracle_donor.append(q_don)
        for t in range(1, a.length):
            if test_conflict[b, t]:
                continue
            difference = max(abs(q_rec[t].get(tok, 0)-q_don[t].get(tok, 0))
                             for tok in test_gen.vocab)
            null_mask[b, t] = difference < 1e-6
    null_model_tag_kl = float((recipient_p*(recipient_logp-donor_logp))
                              .sum(-1)[null_mask].mean())
    reference = odds(donor_logits, test_positions)-odds(recipient_logits, test_positions)
    assert len(reference) > 100
    oracle_indices, oracle_values = [], []
    for index, (b, t, bigram, induction) in enumerate(test_positions):
        big, ind = test_gen.id2token[bigram], test_gen.id2token[induction]
        rec, don = oracle_recipient[b][t], oracle_donor[b][t]
        if min(rec.get(big, 0), rec.get(ind, 0),
               don.get(big, 0), don.get(ind, 0)) <= 0:
            continue
        oracle_indices.append(index)
        oracle_values.append((np.log(don[big])-np.log(don[ind]))
                             -(np.log(rec[big])-np.log(rec[ind])))
    assert len(oracle_indices) > 100
    oracle_shift = torch.tensor(oracle_values, dtype=torch.float32)
    model_shift_rmse_to_oracle = float(
        (reference[oracle_indices]-oracle_shift).square().mean().sqrt())
    reference_power = float(reference.square().mean())
    rows = []
    for name, u in bases.items():
        assert torch.linalg.matrix_norm(u.T@u-torch.eye(config.n_embd)) < 1e-4
        for width in (1, 2, 4, 8, 16, 32, 48, config.n_embd):
            logits = forward_patch(model, bond, x_recipient, donor_acts, u, width,
                                   a.patch_positions)
            patch_logp = F.log_softmax(logits, -1)
            null_patch_kl = float((recipient_p*(recipient_logp-patch_logp))
                                  .sum(-1)[null_mask].mean())
            patched = odds(logits, test_positions)-odds(recipient_logits, test_positions)
            residual = patched-reference
            rows.append(dict(basis=name, width=width,
                             n_conflict_positions=len(reference),
                             target_shift_mean=float(reference.mean()),
                             patched_shift_mean=float(patched.mean()),
                             patched_shift_rmse_to_oracle=float(
                                 (patched[oracle_indices]-oracle_shift)
                                 .square().mean().sqrt()),
                             target_shift_rmse=float(torch.sqrt(residual.square().mean())),
                             residual_power_fraction=float(residual.square().mean()/reference_power),
                             patch_power_fraction=float(patched.square().mean()/reference_power),
                             null_context_patch_kl=null_patch_kl,
                             shift_correlation=float(torch.corrcoef(torch.stack([
                                 reference, patched]))[0, 1])))
    output = dict(config=vars(a), n_fit_conflicts=len(fit_positions),
                  n_test_conflicts=len(test_positions),
                  weight_only_formula_rmse=weight_only_formula_error,
                  contrast_variance_captured=contrast_variance_captured,
                  head_contrast_energy=head_contrast_energy,
                  baseline_no_patch_rmse=float(reference.square().mean().sqrt()),
                  oracle_shift_mean=float(oracle_shift.mean()),
                  n_oracle_positive_conflict_positions=len(oracle_indices),
                  model_shift_rmse_to_oracle=model_shift_rmse_to_oracle,
                  n_oracle_stable_no_conflict_positions=int(null_mask.sum()),
                  null_context_full_tag_kl=null_model_tag_kl,
                  rows=rows,
                  caveat='The weight-only tag contrast enumerates ordinary '
                         'tokens and positions uniformly, without language '
                         'samples; this is a chosen functional metric, not a '
                         'unique canonical gauge. Only one attention-output '
                         'bond is patched. Residual and other '
                         'layers retain recipient tag information; full-width patch need '
                         'not recover donor behavior. No oracle KL is claimed here.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    for row in rows:
        if row['width'] in (4, 16, 64):
            print(row['basis'], row['width'],
                  round(row['target_shift_rmse'], 4),
                  round(row['residual_power_fraction'], 4))


if __name__ == '__main__':
    main()
