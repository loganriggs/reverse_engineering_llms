"""Gauge sensitivity of local QK quotient truncation on trained attention."""

import json

import numpy as np
import torch

from analyze_dual_output_language import factors,score
from analyze_toy_language import env_pair,pair_table
from dual_joint_function_canonical import HERE
from toy_factorial import weighted_trace_gauge
from train_dual_output_language import DualOutputAttention,exhaustive


def relative(approx,truth):
    return float(np.sum((approx-truth)**2)/np.sum(truth**2))


def gauges(seed,head):
    rng=np.random.default_rng(20261117+100*seed+head)
    out={'original':np.eye(2),
         'suppress_axis0':np.diag([.1,1.]),
         'suppress_axis1':np.diag([1.,.1])}
    for j in range(3):
        q,_=np.linalg.qr(rng.standard_normal((2,2)))
        out[f'rotated_scale_{j}']=q@np.diag([.1,1.])@q.T
    return out


def local_key_cut(head,env):
    _,_,vh=np.linalg.svd(head['v'],full_matrices=False)
    p=vh[0]
    direction=np.outer(p,p)
    identity=np.eye(2).reshape(-1)
    flat=direction.reshape(-1)
    alpha=float((flat@env@identity)/(flat@env@flat))
    return alpha*direction


def run():
    torch.set_num_threads(1)
    query_key_measure=np.ones((2,6))/12
    seq_t,target_t,attrs=exhaustive()
    seq=seq_t.numpy();target=target_t.numpy()
    rows=[];tasks=[];gauge_checks=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f'dual_output_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        native=[factors(model,h) for h in range(2)]
        originals=[pair_table(h,np.eye(2)) for h in native]
        bias=model.read.bias.detach().numpy()
        baseline,_=score(originals,seq,target,bias,attrs)
        assert baseline['rare_heldout']['joint_accuracy']==1.
        direct=model(seq_t).detach().numpy()
        assert np.max(np.abs(direct-(bias+sum(np.sum(t[seq[:,-1,None]-6,
                                      seq[:,:-1]],axis=1) for t in originals))))<1e-10
        candidates={}
        for head,source in enumerate(native):
            candidate={}
            for name,g in gauges(seed,head).items():
                inverse=np.linalg.inv(g)
                gauged=dict(source,u=source['u']@g,v=source['v']@inverse.T)
                assert np.max(np.abs(pair_table(gauged,np.eye(2))-
                                         originals[head]))<1e-10
                env=env_pair(gauged,query_key_measure)
                local=local_key_cut(gauged,env)
                wtg,_,_,_=weighted_trace_gauge(env,1)
                result={}
                for method,x in [('local_key_svd',local),('environment_wtg',wtg)]:
                    ordinary=pair_table(gauged,x)
                    cf=dict(gauged,b=np.ones_like(gauged['b']))
                    altered=pair_table(cf,x)
                    reference_cf=pair_table(dict(source,b=np.ones_like(source['b'])),
                                            np.eye(2))
                    result[method]=ordinary
                    rows.append(dict(seed=seed,head=head,gauge=name,
                        gauge_condition=float(np.linalg.cond(g)),method=method,
                        ordinary_relative_mse=relative(ordinary,originals[head]),
                        qk2_constant_relative_mse=relative(altered,reference_cf)))
                candidate[name]=result
            candidates[head]=candidate
            for method in ('local_key_svd','environment_wtg'):
                reference=candidate['original'][method]
                for name in candidate:
                    if name!='original':
                        difference=float(np.max(np.abs(candidate[name][method]-reference)))
                        gauge_checks.append(dict(seed=seed,head=head,method=method,
                                                 gauge=name,function_max_abs_difference=difference))
                        if method=='environment_wtg':
                            assert difference<1e-9
        for gauge in gauges(seed,0):
            for method in ('local_key_svd','environment_wtg'):
                tables=[candidates[h][gauge][method] for h in range(2)]
                result,_=score(tables,seq,target,bias,attrs)
                tasks.append(dict(seed=seed,gauge=gauge,method=method,
                    all_accuracy=result['joint_accuracy'],
                    rare_heldout_accuracy=result['rare_heldout']['joint_accuracy']))
        print('seed',seed,flush=True)
    summary=[]
    for method in ('local_key_svd','environment_wtg'):
        part=[r for r in rows if r['method']==method]
        per_head=[]
        for seed in range(10):
            for head in range(2):
                s=[r for r in part if r['seed']==seed and r['head']==head]
                per_head.append(dict(seed=seed,head=head,
                    ordinary_error_range=max(r['ordinary_relative_mse'] for r in s)-
                                         min(r['ordinary_relative_mse'] for r in s),
                    qk2_error_range=max(r['qk2_constant_relative_mse'] for r in s)-
                                    min(r['qk2_constant_relative_mse'] for r in s)))
        summary.append(dict(method=method,
            mean_ordinary_relative_mse=float(np.mean([r['ordinary_relative_mse'] for r in part])),
            mean_qk2_constant_relative_mse=float(np.mean([r['qk2_constant_relative_mse'] for r in part])),
            median_within_head_ordinary_error_range=float(np.median([x['ordinary_error_range'] for x in per_head])),
            max_within_head_ordinary_error_range=float(max(x['ordinary_error_range'] for x in per_head)),
            mean_rare_heldout_accuracy=float(np.mean([t['rare_heldout_accuracy']
                for t in tasks if t['method']==method]))))
    result=dict(description='Ten trained necessary-two-head toy-language models: exact QK1 quotient GL gauges; compare local key-factor SVD with environment WTG rank-one cuts using uniform output-weighted token pairs, then test ordinary, QK2-constant, and task outcomes',
        qk_quotient_rank=2,physical_qk_width=4,
        gauges_per_head=6,full_function_invariance_gate=1e-10,
        wtg_function_invariance_gate=1e-9,
        fit_uses_task_labels=False,rows=rows,tasks=tasks,
        gauge_checks=gauge_checks,summary=summary)
    (HERE/'dual_trained_qk_quotient_gauge_stress.json').write_text(json.dumps(result,indent=2)+'\n')
    for s in summary:print(s,flush=True)


if __name__=='__main__':run()
