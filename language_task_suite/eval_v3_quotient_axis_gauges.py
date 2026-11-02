"""Rotate a fixed faithful final-residual subspace for axis-level legibility.

Every gauge preserves the rank-k projector exactly. Test whether activation-
or output-weight-sparse rotations make a few axes sufficient for valid edits,
including rule families omitted from rotation fitting.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_v3_rule_holdout_bond import HELD_OUT, orthogonal_columns, panel, remove_duplicates
from eval_v3_two_path_holdout import prepare, readout
from surface_mixture_v3 import SurfaceMixtureV3


def normalized_l1(c):
    norm = c.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    scaled = c/norm
    return (scaled.square()+1e-8).sqrt().sum(-1)


def optimize_rotation(fit, dev, steps, lr):
    d = fit.shape[-1]
    raw = torch.nn.Parameter(torch.eye(d))
    opt = torch.optim.Adam([raw], lr=lr)
    best = (float('inf'),torch.eye(d),0)
    for step in range(steps+1):
        rotation = orthogonal_columns(raw)
        with torch.no_grad():
            dev_loss = normalized_l1(dev@rotation).mean().item()
            if dev_loss < best[0]:
                best = (dev_loss,rotation.detach().clone(),step)
        if step == steps:
            break
        opt.zero_grad()
        normalized_l1(fit@rotation).mean().backward()
        opt.step()
    return best[1],dict(best_dev_l1_l2=best[0],best_step=best[2])


@torch.no_grad()
def score(model, data, u, rotation, widths):
    axes = u@rotation
    coords = data['dr']@axes
    projected_full = coords@axes.T
    full_effect = readout(model,data['r0']+projected_full)-data['p0']
    orig_norm = data['native']
    family_scores = {}
    for kind in sorted(set(data['kinds'])):
        mask = torch.tensor([label == kind for label in data['kinds']])
        c = coords[mask]
        sq = c.square()
        total = sq.sum(-1).clamp_min(1e-12)
        entry = dict(n=int(mask.sum()),
                     effective_axes=(c.abs().sum(-1).square()/total).mean().item(),
                     top1_energy_share=(sq.max(-1).values/total).mean().item(),
                     top4_energy_share=(sq.topk(4,dim=-1).values.sum(-1)/total).mean().item(),
                     full_subspace_to_teacher_l1=(full_effect[mask]-orig_norm[mask]).abs().sum(-1).mean().item())
        for width in widths:
            selected = c.abs().topk(width,dim=-1).indices
            keep = torch.zeros_like(c).scatter_(1,selected,1.0)
            delta = (c*keep)@axes.T
            effect = readout(model,data['r0'][mask]+delta)-data['p0'][mask]
            entry[f'top{width}_to_teacher_l1']=(effect-orig_norm[mask]).abs().sum(-1).mean().item()
            entry[f'top{width}_to_oracle_l1']=(effect-data['oracle'][mask]).abs().sum(-1).mean().item()
        family_scores[kind]=entry
    return family_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir',default='language_task_suite/surface_mixture_v3_object8')
    p.add_argument('--basis-file',required=True)
    p.add_argument('--seed',type=int,default=30101)
    p.add_argument('--test-seed',type=int,default=34001)
    p.add_argument('--streams',type=int,default=160)
    p.add_argument('--fit-cap',type=int,default=120)
    p.add_argument('--dev-cap',type=int,default=60)
    p.add_argument('--test-cap',type=int,default=160)
    p.add_argument('--length',type=int,default=72)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    folder=Path(a.model_dir)
    record=json.loads((folder/'results.json').read_text())['records'][0]
    model=BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     map_location='cpu',weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    saved=torch.load(a.basis_file,map_location='cpu',weights_only=True)
    kind,u=saved['combined_teacher_optimized']
    assert kind=='combined'
    rank=u.shape[1]
    assert rank>=16 and (u.T@u-torch.eye(rank)).abs().max()<1e-5
    vocab=SurfaceMixtureV3(object_vocab_size=8).token2id
    raw_fit=[pair for pair in panel(a.seed,a.streams,a.fit_cap,a.length)
             if pair['kind'] not in HELD_OUT]
    raw_dev=[pair for pair in panel(a.seed+1,a.streams,a.dev_cap,a.length)
             if pair['kind'] not in HELD_OUT]
    raw_test=panel(a.test_seed,a.streams*2,a.test_cap,a.length)
    seen=set()
    fit_pairs,fit_removed=remove_duplicates(raw_fit,seen)
    dev_pairs,dev_removed=remove_duplicates(raw_dev,seen)
    test_pairs,test_removed=remove_duplicates(raw_test,seen)
    assert not any(pair['kind'] in HELD_OUT for pair in fit_pairs+dev_pairs)
    assert all(any(pair['kind']==kind for pair in test_pairs) for kind in HELD_OUT)
    fit=prepare(model,fit_pairs,vocab)
    dev=prepare(model,dev_pairs,vocab)
    test=prepare(model,test_pairs,vocab)
    fit_c=fit['dr']@u
    dev_c=dev['dr']@u
    output_w=model.lm_head.weight@u
    generator=torch.Generator().manual_seed(a.seed+850)
    random=orthogonal_columns(torch.randn(rank,rank,generator=generator))
    _,_,vh=torch.linalg.svd(fit_c.double(),full_matrices=False)
    pca_rotation=vh.T.float().contiguous()
    activation_rotation,activation_meta=optimize_rotation(fit_c,dev_c,a.steps,a.lr)
    output_rotation,output_meta=optimize_rotation(output_w,output_w,a.steps,a.lr)
    rotations=dict(original=torch.eye(rank),random=random,
                   coordinate_pca=pca_rotation,
                   edit_activation_sparse=activation_rotation,
                   output_weight_sparse=output_rotation)
    projector=u@u.T
    gauge_error=max((u@r@(u@r).T-projector).abs().max().item()
                    for r in rotations.values())
    assert gauge_error<1e-5
    widths=(1,2,4,8,16,32) if rank==32 else (1,2,4,8,16)
    results={name:score(model,test,u,rotation,widths)
             for name,rotation in rotations.items()}
    fit_sparsity={name:normalized_l1(fit_c@rotation).mean().item()
                  for name,rotation in rotations.items()}
    output_sparsity={name:normalized_l1(output_w@rotation).mean().item()
                     for name,rotation in rotations.items()}
    result=dict(config=vars(a),rank=rank,heldout_rotation_families=HELD_OUT,
                fit_counts=dict(Counter(pair['kind'] for pair in fit_pairs)),
                dev_counts=dict(Counter(pair['kind'] for pair in dev_pairs)),
                test_counts=dict(Counter(pair['kind'] for pair in test_pairs)),
                duplicates_removed=dict(fit=fit_removed,dev=dev_removed,test=test_removed),
                max_endpoint_replay_error=max(fit['replay_error'],dev['replay_error'],test['replay_error']),
                max_projector_difference=gauge_error,
                optimization=dict(edit_activation=activation_meta,output_weight=output_meta),
                fit_coordinate_l1_l2=fit_sparsity,
                output_weight_l1_l2=output_sparsity,
                results=results,
                caveat='Rotations preserve the rank-k subspace and full patch '
                       'effect. Top-k axes are chosen per edit from a donor delta, '
                       'so this is local counterfactual analysis, not a running '
                       'sparse model. Known edit families select fit contexts; '
                       'induction families are absent from rotation fit/dev.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(rotations,path.with_suffix('.pt'))
    path.write_text(json.dumps(result,indent=2)+'\n')
    for name in rotations:
        print(name,'fit_l1_l2',round(fit_sparsity[name],3),
              'weight_l1_l2',round(output_sparsity[name],3),
              {kind:round(results[name][kind]['top4_to_teacher_l1'],3)
               for kind in HELD_OUT},flush=True)


if __name__=='__main__':
    main()
