"""Exact first-layer source replay and sparse-source causal sufficiency.

Top-K source masks are chosen on the original prefix and held fixed for its
edited counterpart. Ranking is local to a known closing-token contrast; it
is a supervised circuit-selection prior, not task-blind discovery.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_surface_mixture_v2_nested_edits import build_pairs
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen


def source_vectors(model, idx, apply_rotary_emb):
    """One vector per head/key at the final query, after head output mixing."""
    attn = model.blocks[0].attn
    embed = model.embed(idx)
    x = model.blocks[0].norm_attn(embed) if model.use_rmsnorm else embed
    batch, length, width = x.shape
    heads, dim = attn.n_head, attn.head_dim
    project = lambda layer: layer(x).view(batch, length, heads, dim)
    q1, k1, q2, k2, value = [project(layer) for layer in
                              (attn.q1, attn.k1, attn.q2, attn.k2, attn.v)]
    cos, sin = attn.rotary(q1)
    q1, k1 = apply_rotary_emb(q1, cos, sin), apply_rotary_emb(k1, cos, sin)
    q2, k2 = apply_rotary_emb(q2, cos, sin), apply_rotary_emb(k2, cos, sin)
    if attn.qk_norm:
        q1, k1, q2, k2 = [F.normalize(t, dim=-1) for t in
                           (q1, k1, q2, k2)]
    score1 = torch.einsum('bhd,bthd->bht', q1[:, -1], k1)/dim
    score2 = torch.einsum('bhd,bthd->bht', q2[:, -1], k2)/dim
    pattern = score1*score2
    out_weights = attn.out.weight.T.reshape(heads, dim, width)
    vectors = torch.einsum('bht,bthd,hdo->bhto', pattern, value,
                           out_weights)
    return embed[:, -1], pattern, vectors


def logits_from_sources(model, residual, vectors, mask=None):
    if mask is not None:
        vectors = vectors*mask[..., None]
    x = residual+vectors.sum(dim=(1, 2))
    if model.use_rmsnorm or model.use_final_norm:
        x = model.final_norm(x)
    return model.lm_head(x)


def single_source_kl(model, residual, vectors, full_logits):
    """Teacher-output KL after deleting each head/source in isolation."""
    heads, length, width = vectors.shape
    reduced = (residual+vectors.sum(dim=(0,1))-
               vectors.reshape(heads*length,width))
    if model.use_rmsnorm or model.use_final_norm:
        reduced = model.final_norm(reduced)
    logp = F.log_softmax(model.lm_head(reduced).float(),dim=-1)
    p = F.softmax(full_logits.float(),dim=-1)
    impact = (p*(p.clamp_min(1e-30).log()-logp)).sum(-1)
    return impact.reshape(heads,length)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['method'], row['k'], row['kind'])].append(row)
        grouped[(row['method'], row['k'],
                 row['kind']+f"@lag{row['lag']}")].append(row)
    return [dict(method=method, k=k, kind=kind, n=len(items),
                 **{key: float(np.mean([item[key] for item in items]))
                    for key in ('teacher_kl', 'teacher_effect_l1',
                                'oracle_effect_l1', 'after_correct',
                                'opener_selected')})
            for (method, k, kind), items in sorted(grouped.items())]


def verb_cases(gen, nstreams, length, limit):
    cases = []
    for _ in range(nstreams):
        ids, labels, qs = gen.sample_sequence(length)
        for t, label in enumerate(labels):
            if label == 'mixed_verb':
                cases.append((ids[:t], {gen.token2id[token]: mass
                                        for token, mass in qs[t].items()}))
                if len(cases) >= limit:
                    return cases
    return cases


@torch.no_grad()
def evaluate_verbs(model, cases, apply_rotary_emb, ks, device, seed):
    rng = torch.Generator(device=device).manual_seed(seed)
    rows = defaultdict(list)
    native_kls = []
    for prefix, q in cases:
        idx = torch.tensor([prefix], dtype=torch.long, device=device)
        residual, pattern, vectors = source_vectors(model, idx,
                                                     apply_rotary_emb)
        native_logits = model(idx)[0][0, -1]
        native = F.softmax(native_logits.float(), dim=-1)
        oracle = torch.zeros_like(native)
        for token, mass in q.items():
            oracle[token] = mass
        oracle_log = oracle.clamp_min(1e-30).log()
        kl_to = lambda p: float((oracle*(oracle_log-
                    p.clamp_min(1e-30).log())).sum())
        native_kls.append(kl_to(native))
        rankers = {
            'score_abs': pattern[0].abs(),
            'vector_norm': vectors[0].norm(dim=-1),
            'single_source_kl': single_source_kl(model,residual[0],
                                                  vectors[0],native_logits),
            'random': torch.rand(pattern[0].shape, generator=rng,
                                 device=device),
        }
        for method, ranking in rankers.items():
            for k in ks:
                chosen = ranking.topk(min(k, idx.shape[1]),dim=-1).indices
                mask = torch.zeros_like(ranking,dtype=torch.bool)
                mask.scatter_(1,chosen,True)
                sparse = F.softmax(logits_from_sources(model,residual,vectors,
                    mask[None]).float()[0],dim=-1)
                teacher_kl = float((native*(native.clamp_min(1e-30).log()-
                                   sparse.clamp_min(1e-30).log())).sum())
                rows[(method,k)].append((kl_to(sparse),teacher_kl))
    return dict(n=len(cases),native_oracle_kl=float(np.mean(native_kls)),
        rows=[dict(method=method,k=k,oracle_kl=float(np.mean(v,axis=0)[0]),
                   teacher_kl=float(np.mean(v,axis=0)[1]))
              for (method,k),v in sorted(rows.items())])


@torch.no_grad()
def evaluate(model, pairs, apply_rotary_emb, ks, device, seed):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    rows = []
    replay_max_abs = 0.
    for pair in pairs:
        idx = torch.tensor([pair['original'], pair['edited']],
                           dtype=torch.long, device=device)
        residual, pattern, vectors = source_vectors(model, idx,
                                                     apply_rotary_emb)
        full = logits_from_sources(model, residual, vectors)
        native = model(idx)[0][:, -1]
        replay_max_abs = max(replay_max_abs,
                             float((full-native).abs().max()))
        baseline = F.softmax(native.float(), dim=-1)
        before_answer, after_answer = (pair['answer_before'],
                                        pair['answer_after'])
        oracle_delta = torch.zeros_like(baseline[0])
        oracle_delta[after_answer] += 1
        oracle_delta[before_answer] -= 1
        contrast = (model.lm_head.weight[after_answer]-
                    model.lm_head.weight[before_answer])
        rankers = {
            'score_abs': pattern[0].abs(),
            'vector_norm': vectors[0].norm(dim=-1),
            'single_source_kl': single_source_kl(model,residual[0],
                                                  vectors[0],native[0]),
            'closer_contrast': torch.einsum('htd,d->ht', vectors[0],
                                            contrast).abs(),
            'random': torch.rand(pattern[0].shape, generator=rng,
                                 device=device),
        }
        length = idx.shape[1]
        for method, ranking in rankers.items():
            for k in ks:
                count = min(k, length)
                chosen = ranking.topk(count, dim=-1).indices
                mask = torch.zeros_like(ranking, dtype=torch.bool)
                mask.scatter_(1, chosen, True)
                sparse = F.softmax(logits_from_sources(
                    model, residual, vectors, mask[None].expand(2,-1,-1)
                    ).float(), dim=-1)
                delta = sparse[1]-sparse[0]
                teacher_delta = baseline[1]-baseline[0]
                kl = (baseline*(baseline.clamp_min(1e-30).log()-
                      sparse.clamp_min(1e-30).log())).sum(-1).mean()
                opener_pos = length-pair['opener_distance']
                rows.append(dict(method=method,k=k,kind=pair['kind'],
                    lag=pair['opener_distance'],
                    teacher_kl=float(kl),
                    teacher_effect_l1=float((delta-teacher_delta).abs().sum()),
                    oracle_effect_l1=float((delta-oracle_delta).abs().sum()),
                    after_correct=float(sparse[1].argmax()==after_answer),
                    opener_selected=float(mask[:,opener_pos].float().mean())))
    return dict(replay_max_abs=replay_max_abs, rows=summarize(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--nstreams', type=int, default=256)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261204)
    p.add_argument('--ks', default='0,1,2,4,8,16')
    p.add_argument('--verb-streams', type=int, default=256)
    p.add_argument('--verb-limit', type=int, default=512)
    p.add_argument('--eval-extra-mass', type=float)
    p.add_argument('--eval-inner-mass', type=float)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config, apply_rotary_emb
    torch.set_num_threads(4)
    meta = json.loads((args.model_dir/'results.json').read_text())
    assert meta['config']['generator']=='v2_nested_varlen'
    record = next(row for row in meta['records'] if row['depth']==1)
    assert record['model_config']['block_has_mlp']==[False]
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(args.model_dir/'model_depth1.pt',
                                     map_location='cpu',weights_only=True))
    model = model.to(args.device)
    generator_kwargs=dict(meta['generator_kwargs'])
    if args.eval_extra_mass is not None:
        generator_kwargs['extra_mass']=args.eval_extra_mass
    if args.eval_inner_mass is not None:
        generator_kwargs['inner_mass']=args.eval_inner_mass
    gen = SurfaceMixtureV2NestedVarlen(seed=args.seed,
                                      **generator_kwargs)
    pairs = build_pairs(gen,args.nstreams,args.length)
    ks=[int(x) for x in args.ks.split(',')]
    result = evaluate(model,pairs,apply_rotary_emb,ks,args.device,
                      args.seed+1)
    assert result['replay_max_abs'] < 5e-4, result['replay_max_abs']
    result['config']={**vars(args),'model_dir':str(args.model_dir),
                      'output':str(args.output)}
    result['pair_counts']={kind:sum(pair['kind']==kind for pair in pairs)
                           for kind in sorted(set(p['kind'] for p in pairs))}
    verb_gen = SurfaceMixtureV2NestedVarlen(seed=args.seed+100,
                                           **generator_kwargs)
    result['verbs'] = evaluate_verbs(model,verb_cases(verb_gen,
        args.verb_streams,args.length,args.verb_limit),apply_rotary_emb,
        ks,args.device,args.seed+101)
    result['caveat']='Masks use the original prefix and are held fixed '
    result['caveat']+='under edits. Closer-contrast selection uses known '
    result['caveat']+='answer labels. This tests local source sufficiency, '
    result['caveat']+='not a unique circuit or task-blind discovery.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('replay max abs',result['replay_max_abs'],'pairs',len(pairs))
    for row in result['rows']:
        if row['kind']=='outer_relevant' and row['k'] in (1,4,16):
            print(row['method'],row['k'],'teacher KL',
                  round(row['teacher_kl'],4),'oracle edit L1',
                  round(row['oracle_effect_l1'],4),'opener selected',
                  round(row['opener_selected'],3))


if __name__=='__main__':
    main()
