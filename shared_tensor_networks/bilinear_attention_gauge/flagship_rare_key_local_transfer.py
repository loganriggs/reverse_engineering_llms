"""Broad exact first-layer output check for rare-key metric transfer.

Use 128 disjoint documents per rare/common stratum and corpus. Compare
uniform/mixed/corpus rank-96 pre-RoPE QK1 cuts on only the selected key's
causal contribution after the native OV/output projection. Cross-check
against saved full-model hook energies on the 24-case panel.
"""

import json
import argparse

import numpy as np
import torch

from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project,rotate,rotary
from flagship_ood_corpora import DATASETS
from flagship_rare_key_metric_transfer import (
    LENGTH,PRIORS,TARGET_POSITION,frequency_counts,select_windows)
from flagship_transfer import ART


def local_rows(state,cases,cuts):
    token_ids=np.unique(np.concatenate([np.asarray(c['ids'][:LENGTH],dtype=np.int64)
                                        for c in cases]))
    projected=project(state,token_ids,0,('c_q','c_k','c_q2','c_k2','c_v'))
    lookup={int(token):i for i,token in enumerate(token_ids)}
    output_weight=state['transformer.h.0.attn.c_proj.weight'][:,:HD].to(
        torch.float64).numpy()
    cos,sin=rotary(LENGTH)
    positions=np.arange(LENGTH)
    records=[]
    for index,case in enumerate(cases):
        inv=np.array([lookup[int(t)] for t in case['ids'][:LENGTH]])
        j=case['key_position']
        qpre=projected['c_q'][inv]
        qpost=rotate(qpre,positions,cos,sin)
        kpost=rotate(projected['c_k'][inv[j:j+1]],
                     np.array([j]),cos,sin)[0]
        q2post=rotate(projected['c_q2'][inv],positions,cos,sin)
        k2post=rotate(projected['c_k2'][inv[j:j+1]],
                      np.array([j]),cos,sin)[0]
        score1=(qpost@kpost)/HD
        score2=(q2post@k2post)/HD
        value=projected['c_v'][inv[j]]@output_weight.T
        value_energy=float(value@value)
        native=(score1*score2)
        native[:j]=0.
        native_energy=float(np.sum(native*native)*value_energy)
        assert native_energy>=0
        for name,cut in cuts.items():
            cut_qpost=rotate(qpre@cut,positions,cos,sin)
            score1_cut=(cut_qpost@kpost)/HD
            delta=(score1_cut-score1)*score2
            delta[:j]=0.
            removed_energy=float(np.sum(delta*delta)*value_energy)
            records.append(dict(case_index=index,prior=name,
                native_key_output_energy=native_energy,
                removed_post_c_proj_energy=removed_energy,
                relative_key_output_error=removed_energy/max(native_energy,1e-30)))
    return records


def check_full_hook(state,cuts,full):
    cases=full['cases'][:12]
    local=local_rows(state,cases,cuts)
    lookup={(r['case_index'],r['prior']):r for r in full['records']}
    errors=[]
    for r in local:
        actual=lookup[r['case_index'],r['prior']]['removed_post_c_proj_energy']
        predicted=r['removed_post_c_proj_energy']
        errors.append(abs(predicted-actual)/max(actual,1e-12))
    maximum=float(np.max(errors))
    assert maximum<1e-3,maximum
    return maximum


def run(n=128,seed=20264150,fit_seed=41):
    torch.set_num_threads(4)
    counts=frequency_counts()
    corpora={'fineweb_test':(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    cases=[]
    for index,(name,(path,begin,end)) in enumerate(corpora.items()):
        cases.extend(select_windows(name,path,begin,end,counts,n,seed+index))
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    arrays=np.load(ART/f'head0_rope_metric_factorial_seed{fit_seed}.npz')
    cuts={name:arrays[f'{name}__pre'] for name in PRIORS}
    hook_gate=None
    if fit_seed==41:
        full=json.loads((ART/'head0_pre_rank96_rare_key_transfer_n24_s20264100.json').read_text())
        hook_gate=check_full_hook(state,cuts,full)
    records=local_rows(state,cases,cuts)
    result=dict(description='Exact layer-0/head-0 selected-key contribution after c_proj for uniform/mixed/corpus pre-RoPE rank-96 cuts. 128 disjoint rare and common natural documents per corpus selected by FineWeb train token frequency; all other key contributions cancel. This is an immediate-output metric, not full-model KL. Twelve full-model panel cases cross-check the score/OV calculation against the actual hook output energy.',
                n_per_stratum_per_corpus=n,seed=seed,length=LENGTH,
                key_position=TARGET_POSITION,fit_cut_seed=fit_seed,
                full_hook_max_relative_energy_error=hook_gate,
                cases=cases,records=records)
    (ART/f'head0_pre_rank96_rare_key_local_fit{fit_seed}_n{n}_s{seed}.json').write_text(
        json.dumps(result,indent=2)+'\n')
    for corpus in corpora:
        for stratum in ('rare','common'):
            indices={i for i,c in enumerate(cases)
                     if c['corpus']==corpus and c['stratum']==stratum}
            part=[r for r in records if r['case_index'] in indices]
            print(corpus,stratum,{name:round(sum(r['removed_post_c_proj_energy']
                for r in part if r['prior']==name)/sum(r['native_key_output_energy']
                for r in part if r['prior']==name),4) for name in PRIORS},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=128)
    p.add_argument('--seed',type=int,default=20264150)
    p.add_argument('--fit-seed',type=int,default=41)
    a=p.parse_args()
    run(a.n,a.seed,a.fit_seed)
