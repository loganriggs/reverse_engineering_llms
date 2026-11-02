"""Penalize the exact first-layer vocabulary QK product, not matrix entries.

This is a deliberately narrow first-layer functional sparsity probe. The penalty
averages over all token pairs at several relative positions, either uniformly
or under separately sampled training-pair frequencies. Evaluation is separate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_varied_first_layer_fold import top_abs_mass
from surface_mixture_v2_varied import SurfaceMixtureV2Varied
from train_surface_mixture import evaluate, make_eval


def functional_score_tables(model, lags=(1, 4, 16)):
    from model import apply_rotary_emb

    block = model.blocks[0]
    attn = block.attn
    emb = block.norm_attn(model.embed.weight)
    h, d = attn.n_head, attn.head_dim
    projected = {name: getattr(attn, name)(emb).reshape(-1, 1, h, d)
                 for name in ('q1', 'k1', 'q2', 'k2')}
    dummy = projected['q1'][:, :1].expand(-1, max(lags)+1, -1, -1)
    cos, sin = attn.rotary(dummy)
    tables = []
    for lag in lags:
        branch_scores = []
        for branch in (1, 2):
            q = apply_rotary_emb(projected[f'q{branch}'],
                                 cos[:, lag:lag+1], sin[:, lag:lag+1])[:, 0]
            k = projected[f'k{branch}'][:, 0]  # RoPE at position zero.
            branch_scores.append(torch.einsum('vhd,uhd->hvu', q, k)/d)
        tables.append(branch_scores[0]*branch_scores[1])
    return tables


def stats(model):
    with torch.no_grad():
        result = {}
        for lag, table in zip((1, 4, 16), functional_score_tables(model)):
            scale = table.abs().mean().clamp_min(1e-30)
            result[str(lag)] = dict(mean_abs=float(scale),
                                    l1_over_rms=float(scale/table.square().mean().sqrt()),
                                    top_one_percent_abs_mass=top_abs_mass(table),
                                    fraction_below_one_percent_mean=float(
                                        (table.abs() < .01*scale).float().mean()))
        return result


def observed_pair_weights(kwargs, lags, streams, length, seed, vocab_size):
    generator = SurfaceMixtureV2Varied(seed=seed, **kwargs)
    counts = [np.zeros((vocab_size, vocab_size), dtype=np.int64)
              for _ in lags]
    for start in range(0, streams, 512):
        ids = np.asarray(generator.sample_batch(min(512, streams-start),
                                                length)[0])
        for lag, count in zip(lags, counts):
            np.add.at(count, (ids[:, lag:].reshape(-1),
                              ids[:, :-lag].reshape(-1)), 1)
    return [torch.from_numpy(count.astype(np.float32)/count.sum())
            for count in counts], [int((count > 0).sum()) for count in counts]


def weighted_moments(table, weights):
    if weights is None:
        return table.abs().mean(), table.square().mean().sqrt()
    return (table.abs()*weights).sum((-1, -2)).mean(), (
        (table.square()*weights).sum((-1, -2)).mean()).sqrt()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path,
                   default=Path('language_task_suite/surface_mixture_v2_varied_soft_models'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--penalty-weight', type=float, default=.03)
    p.add_argument('--penalty-kind', choices=('mean_abs', 'l1_over_rms'),
                   default='mean_abs')
    p.add_argument('--pair-weighting', choices=('uniform', 'observed'),
                   default='uniform')
    p.add_argument('--count-streams', type=int, default=8192)
    p.add_argument('--count-seed', type=int, default=20261201)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    saved = json.loads((a.model_dir/'results.json').read_text())
    assert saved['config']['generator'] == 'v2_varied'
    record = next(row for row in saved['records'] if row['depth'] == 2)
    kwargs = saved['generator_kwargs']
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     map_location='cpu', weights_only=True))
    model.eval()
    original_stats = stats(model)
    lags = (1, 4, 16)
    if a.pair_weighting == 'observed':
        weights, observed_pair_counts = observed_pair_weights(
            kwargs, lags, a.count_streams, a.length, a.count_seed,
            model.config.vocab_size)
    else:
        weights, observed_pair_counts = [None]*len(lags), None
    initial_tables = functional_score_tables(model)
    initial_moments = [weighted_moments(x.detach(), w)
                       for x, w in zip(initial_tables, weights)]
    baseline_scales = [mean.clamp_min(1e-12)
                       for mean, _ in initial_moments]
    baseline_ratios = [mean/rms.clamp_min(1e-12)
                       for mean, rms in initial_moments]
    train_gen = SurfaceMixtureV2Varied(seed=9100+a.seed, **kwargs)
    selection = make_eval(SurfaceMixtureV2Varied(seed=20261125, **kwargs),
                          512, a.length, 'cpu')
    fresh = make_eval(SurfaceMixtureV2Varied(seed=20261126, **kwargs),
                      512, a.length, 'cpu')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    checkpoint = a.output_dir/'train_checkpoint.pt'
    protocol = dict(steps=a.steps, batch=a.batch, length=a.length, lr=a.lr,
                    penalty_weight=a.penalty_weight,
                    penalty_kind=a.penalty_kind,
                    pair_weighting=a.pair_weighting,
                    count_streams=a.count_streams, count_seed=a.count_seed,
                    seed=a.seed)
    start_step = 0
    if a.resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['protocol'] != protocol:
            raise ValueError('Resume protocol differs from checkpoint')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        train_gen.rng.setstate(state['generator_rng_state'])
        start_step = state['next_step']
        print('resumed at step', start_step, flush=True)
    model.train()
    for step in range(start_step, a.steps):
        panel = make_eval(train_gen, a.batch, a.length, 'cpu')
        logits = model(panel['x'])[0].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               panel['y'].reshape(-1))
        logp = F.log_softmax(logits, -1)
        loss += 2.*(-(panel['q']*logp).sum(-1))[panel['verb']].mean()
        tables = functional_score_tables(model)
        moments = [weighted_moments(table, weight)
                   for table, weight in zip(tables, weights)]
        if a.penalty_kind == 'mean_abs':
            terms = [mean/scale for (mean, _), scale in
                     zip(moments, baseline_scales)]
        else:
            terms = [(mean/rms.clamp_min(1e-12))/ratio
                     for (mean, rms), ratio in zip(moments, baseline_ratios)]
        penalty = sum(terms)/len(terms)
        loss += a.penalty_weight*penalty
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite functional sparsity loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        opt.step()
        if (step+1) % 50 == 0:
            print('step', step+1, 'loss', float(loss.detach()),
                  'penalty', float(penalty.detach()),
                  flush=True)
        if (step+1) % 25 == 0:
            state = dict(protocol=protocol, model=model.state_dict(),
                         optimizer=opt.state_dict(),
                         generator_rng_state=train_gen.rng.getstate(),
                         next_step=step+1)
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(checkpoint)
    model.eval()
    def kl(panel):
        measured = evaluate(model, panel)
        return {key: measured[key]['kl'] for key in
                ('all', 'verb', 'verb_with_induction')}
    output = dict(config={**vars(a), 'source': str(a.source),
                          'model_dir': str(a.model_dir),
                          'output_dir': str(a.output_dir)},
                  original_stats=original_stats, final_stats=stats(model),
                  observed_pair_counts=observed_pair_counts,
                  selection=kl(selection), fresh=kl(fresh),
                  caveat='First-layer pair penalty at lags 1/4/16; '
                         'it does not establish a sparse causal rule path.')
    model_tmp = a.output_dir/'model_depth2.tmp'
    torch.save(model.state_dict(), model_tmp)
    model_tmp.replace(a.output_dir/'model_depth2.pt')
    results = json.dumps(dict(
        config={'generator': 'v2_varied'}, generator_kwargs=kwargs,
        records=[{'depth': 2, 'model_config': record['model_config']}]), indent=2)+'\n'
    (a.output_dir/'results.tmp').write_text(results)
    (a.output_dir/'results.tmp').replace(a.output_dir/'results.json')
    (a.output_dir/'functional_sparsity_results.tmp').write_text(
        json.dumps(output, indent=2)+'\n')
    (a.output_dir/'functional_sparsity_results.tmp').replace(
        a.output_dir/'functional_sparsity_results.json')
    checkpoint.unlink(missing_ok=True)
    print(json.dumps({k: output[k] for k in
                      ('selection', 'fresh', 'final_stats')}, indent=2))


if __name__ == '__main__':
    main()
