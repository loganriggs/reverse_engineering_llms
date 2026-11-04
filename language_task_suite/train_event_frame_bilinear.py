"""Paired fixed/variable-frame bilinear LM training with oracle action gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from event_frame_oracle import ACTION, ATTRIBUTES, PASSIVE, EventFrameOracle, GENRES
from train_corpus_delimiter_bilinear import _model


def pack(samples, generator, n_ctx, with_sites):
    n = len(samples)
    eos = generator.token2id['<eos>']
    x = torch.full((n, n_ctx), eos, dtype=torch.long)
    y = torch.full((n, n_ctx), -100, dtype=torch.long)
    docs, positions, oracle, ambiguous = [], [], [], []
    for index, sample in enumerate(samples):
        ids = [generator.token2id[token] for token in sample.tokens]
        if len(ids) - 1 > n_ctx:
            raise ValueError(f'Document {index} exceeds context {n_ctx}')
        x[index, :len(ids)-1] = torch.tensor(ids[:-1])
        y[index, :len(ids)-1] = torch.tensor(ids[1:])
        if with_sites:
            for site in sample.actions:
                q = generator.next_distribution(
                    sample.tokens[:site.position])
                assert site.answer in q
                row = torch.zeros(len(generator.vocab))
                for token, prob in q.items():
                    row[generator.token2id[token]] = prob
                docs.append(index)
                positions.append(site.position-1)
                oracle.append(row)
                ambiguous.append(len(q) > 1)
    result = {'x': x, 'y': y}
    if with_sites:
        result.update(site_doc=torch.tensor(docs),
                      site_position=torch.tensor(positions),
                      site_oracle=torch.stack(oracle),
                      site_ambiguous=torch.tensor(ambiguous))
    return result


def make_cache(path, train_n, test_n, seed, n_ctx):
    identity = (train_n, test_n, seed, n_ctx)
    if path.exists():
        cached = torch.load(path, map_location='cpu', weights_only=True)
        if tuple(cached['identity']) != identity:
            raise ValueError('Training cache identity mismatch')
        return cached
    data = {'identity': identity, 'arms': {}}
    for split, count, split_seed in (('train', train_n, seed),
                                     ('test', test_n, seed + 100000)):
        gens = {name: EventFrameOracle(seed=split_seed,
                                      variable_frames=(name == 'variable'))
                for name in ('variable', 'fixed')}
        samples = {name: [] for name in gens}
        for _ in range(count):
            left = gens['variable'].sample_document()
            right = gens['fixed'].sample_document()
            assert left.genre == right.genre and left.events == right.events
            samples['variable'].append(left)
            samples['fixed'].append(right)
        for name, gen in gens.items():
            data['arms'].setdefault(name, {})[split] = pack(
                samples[name], gen, n_ctx, with_sites=(split == 'test'))
        print('built', split, count, flush=True)
    data['vocab_size'] = len(gens['variable'].vocab)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(data, temp)
    temp.replace(path)
    return data


def causal_pairs(variable_frames, n_ctx):
    gen = EventFrameOracle(seed=0, variable_frames=variable_frames)
    left, right, target, oracle_delta = [], [], [], []
    pairs = (('clean', 'dusty'), ('broken', 'rare'),
             ('valuable', 'heavy'))
    for obj in gen.objects[:8]:
        for person in gen.people[:2]:
            for before, after in pairs:
                for fact_frame in range(3 if variable_frames else 1):
                    for action_frame in range(3 if variable_frames else 1):
                        state = ((obj, before),)
                        facts = [row for row in gen.candidates((), GENRES[0])
                                 if row.kind == 'fact' and row.obj == obj and
                                 row.attribute == before and
                                 row.frame == fact_frame]
                        acts = [row for row in gen.candidates(state, GENRES[0])
                                if row.kind == 'action' and row.obj == obj and
                                row.person == person and
                                row.frame == action_frame]
                        assert len(facts) == len(acts) == 1
                        fact, action = facts[0], acts[0]
                        prefix = fact.tokens + action.tokens[:action.action_position]
                        edited_fact = tuple(after if token == before else token
                                            for token in fact.tokens)
                        edited = edited_fact + action.tokens[:action.action_position]
                        q0, q1 = gen.next_distribution(prefix), gen.next_distribution(edited)
                        new_action = (PASSIVE[after] if action_frame == 1
                                      else ACTION[after])
                        assert q1.get(new_action, 0.) > q0.get(new_action, 0.)
                        left.append(prefix)
                        right.append(edited)
                        target.append(gen.token2id[new_action])
                        oracle_delta.append(q1.get(new_action, 0.)-
                                            q0.get(new_action, 0.))
    eos = gen.token2id['<eos>']
    def pad(rows):
        x = torch.full((len(rows), n_ctx), eos, dtype=torch.long)
        pos = []
        for i, row in enumerate(rows):
            ids = [gen.token2id[token] for token in row]
            assert len(ids) <= n_ctx
            x[i, :len(ids)] = torch.tensor(ids)
            pos.append(len(ids)-1)
        return x, torch.tensor(pos)
    x0, p0 = pad(left)
    x1, p1 = pad(right)
    return {'x0': x0, 'p0': p0, 'x1': x1, 'p1': p1,
            'target': torch.tensor(target),
            'oracle_delta': torch.tensor(oracle_delta)}


@torch.no_grad()
def evaluate(model, data, pairs, device, batch_size):
    model.eval()
    full_nll, full_tokens = 0., 0
    site_kl, site_nll, site_count = 0., 0., 0
    site_kl_ambig, n_ambig = 0., 0
    for start in range(0, len(data['x']), batch_size):
        end = min(len(data['x']), start + batch_size)
        x, y = data['x'][start:end].to(device), data['y'][start:end].to(device)
        logits = model(x).float()
        full_nll += F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                     ignore_index=-100,
                                     reduction='sum').item()
        full_tokens += (y != -100).sum().item()
        mask = ((data['site_doc'] >= start) &
                (data['site_doc'] < end))
        if not mask.any():
            continue
        indices = data['site_doc'][mask].to(device) - start
        positions = data['site_position'][mask].to(device)
        oracle = data['site_oracle'][mask].to(device)
        logp = F.log_softmax(logits[indices, positions], -1)
        entropy = -(oracle*oracle.clamp_min(1e-30).log()).sum(-1)
        ce = -(oracle*logp).sum(-1)
        kl = ce-entropy
        site_kl += kl.sum().item()
        site_nll += ce.sum().item()
        site_count += len(kl)
        ambig = data['site_ambiguous'][mask].to(device)
        site_kl_ambig += kl[ambig].sum().item()
        n_ambig += ambig.sum().item()
    delta, error, positive, count = 0., 0., 0, 0
    for start in range(0, len(pairs['target']), batch_size):
        end = min(len(pairs['target']), start + batch_size)
        a = model(pairs['x0'][start:end].to(device)).float()
        b = model(pairs['x1'][start:end].to(device)).float()
        ix = torch.arange(end-start, device=device)
        target = pairs['target'][start:end].to(device)
        p0 = F.softmax(a[ix, pairs['p0'][start:end].to(device)], -1)[ix,target]
        p1 = F.softmax(b[ix, pairs['p1'][start:end].to(device)], -1)[ix,target]
        effect = p1-p0
        expected = pairs['oracle_delta'][start:end].to(device)
        delta += effect.sum().item()
        error += (effect-expected).abs().sum().item()
        positive += (effect > 0).sum().item()
        count += len(effect)
    return {'full_token_ce': full_nll/full_tokens,
            'valid_tokens': full_tokens,
            'action_sites': site_count,
            'action_oracle_ce': site_nll/site_count,
            'action_oracle_kl': site_kl/site_count,
            'ambiguous_action_sites': n_ambig,
            'ambiguous_action_kl': site_kl_ambig/n_ambig if n_ambig else None,
            'causal_pairs': count,
            'causal_positive_fraction': positive/count,
            'causal_mean_effect': delta/count,
            'causal_effect_mae': error/count,
            'oracle_mean_effect': pairs['oracle_delta'].mean().item()}


def save_state(path, model, optimizer, rng, next_step, history, identity):
    state = {'identity': identity, 'next_step': next_step,
             'model': {k: v.detach().cpu() for k,v in model.state_dict().items()},
             'optimizer': optimizer.state_dict(), 'rng': rng.get_state(),
             'history': history,
             'cuda_rng': torch.cuda.get_rng_state_all()
                         if next(model.parameters()).is_cuda else []}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-documents', type=int, default=20000)
    p.add_argument('--test-documents', type=int, default=1000)
    p.add_argument('--data-seed', type=int, default=42801)
    p.add_argument('--n-ctx', type=int, default=80)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=42811)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    cache = make_cache(a.cache, a.train_documents, a.test_documents,
                       a.data_seed, a.n_ctx)
    report = {'config': {k: str(v) if isinstance(v, Path) else v
                         for k,v in vars(a).items()},
              'vocab_size': cache['vocab_size'], 'results': {},
              'scope': 'Fixed/variable arms have paired latent semantic '
                       'events and genres, independent train/test streams. '
                       'All next-token positions train with equal CE; no '
                       'site weighting. Action KL uses each arm own '
                       'visible-prefix oracle. Supported edits span all '
                       'available fact/action frames but only a subset of '
                       'objects/people/attribute pairs. This is a diagnostic '
                       'grammar, not a realistic text DGP.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for arm in ('fixed', 'variable'):
        train = cache['arms'][arm]['train']
        test = cache['arms'][arm]['test']
        pairs = causal_pairs(arm == 'variable', a.n_ctx)
        torch.manual_seed(a.seed)
        model = _model(cache['vocab_size'], a.n_ctx, a.depth,
                       a.d_model, a.heads, a.device, 'rms')
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
        rng = torch.Generator().manual_seed(a.seed+1000)
        path = a.state_dir/f'{arm}_depth{a.depth}_seed{a.seed}.pt'
        identity = (arm, a.batch_size, a.lr, a.seed,
                    a.depth, a.d_model, a.heads, tuple(cache['identity']))
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            previous_identity = tuple(saved['identity'])
            if len(previous_identity) == len(identity) + 1:
                previous_identity = previous_identity[:1] + previous_identity[2:]
            if previous_identity != identity or saved['next_step'] > a.steps:
                raise ValueError('Resume identity mismatch')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            rng.set_state(saved['rng'])
            if a.device.startswith('cuda'):
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            history, start = saved['history'], saved['next_step']
            print('resumed', arm, start, flush=True)
        else:
            history = {'0': evaluate(model, test, pairs, a.device,
                                     a.eval_batch_size)}
            start = 0
        last_loss = None
        for step in range(start, a.steps):
            model.train()
            idx = torch.randint(len(train['x']), (a.batch_size,), generator=rng)
            x = train['x'][idx].to(a.device)
            y = train['y'][idx].to(a.device)
            logits = model(x).float()
            loss = F.cross_entropy(logits.flatten(0,1), y.flatten(),
                                   ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss {arm} step {step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            last_loss = float(loss.detach())
            completed = step+1
            if completed % a.eval_every == 0 or completed == a.steps:
                history[str(completed)] = evaluate(
                    model, test, pairs, a.device, a.eval_batch_size)
                history[str(completed)]['train_loss_last'] = last_loss
                report['results'][arm] = {'parameters': model.n_params,
                                          'history': history}
                a.output.write_text(json.dumps(report, indent=2) + '\n')
                save_state(path, model, optimizer, rng, completed,
                           history, identity)
                print(arm, completed, 'loss', round(last_loss, 4),
                      'action KL', round(history[str(completed)][
                          'action_oracle_kl'], 4), flush=True)
        report['results'][arm] = {'parameters': model.n_params,
                                  'history': history}
        a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
