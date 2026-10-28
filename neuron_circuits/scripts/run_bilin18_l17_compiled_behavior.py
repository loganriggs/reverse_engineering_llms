#!/usr/bin/env python3
"""Test fixed compiled MLP programs through final normalization and vocabulary head."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bilin18_compiled_program import load_program,forward_normalized
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_role_compression import B
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_compiled_behavior'
@torch.inference_mode()
def main():
    global STEM
    factorial='--factorial' in sys.argv
    if factorial:STEM=ART/'bilin18_l17_compilation_error_factorial'
    torch.set_num_threads(6);spec=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());cfg=json.loads((ART/'bilin18_l17_compiled_partners.json').read_text())['parent_config'];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,hw,_=collect_sources(model,corpus,{**cfg,'layer':17});targets=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).reshape(-1)).cuda();x=F.rms_norm(pre,(1152,));write=B(x,x,L,R,D)
    dc,_,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'layer':17});dx=F.rms_norm(dc,(1152,));meanwrite=B(dx,dx,L,R,D).mean(0);rows={};counts={}
    if factorial:
        from run_bilin18_l8_context_role_compression import state_rules
        f=np.load(ART/'bilin18_depth_linear_context_frozen.npz');mu=torch.from_numpy(f['L17_mean']).cuda();p=torch.from_numpy(f['L17_pca'][:,:16]).cuda()
        v=torch.from_numpy(np.load(ART/'bilin18_depth_source_geometry_frozen.npz')['L17_global'][:16]).cuda()
        crossbias=meanwrite-state_rules(dx,mu,p.T,p,L,R,D)['cross'].mean(0)
        compact=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda')
        programs={'context':None,'output':None,'both':compact}
        def evaluate(name,z):
            if name=='context':return state_rules(z,mu,p.T,p,L,R,D)['cross']+crossbias
            if name=='output':return meanwrite+((B(z,z,L,R,D)-meanwrite)@v.T)@v
            return forward_normalized(z,compact)
    else:
        programs={m:load_program(ART/f'bilin18_l17_compiled_partners_m{m}_factorized.npz','cuda') for m in spec['programs']}
        def evaluate(name,z):return forward_normalized(z,programs[name])
    pred={m:evaluate(m,x) for m in programs}
    def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
    def accumulate(key,values):
        rows.setdefault(key,np.zeros(cfg['rows']));rows[key]+=values.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
    def measure_states(states,kind):
        metrics={}
        for off in range(0,len(pre),spec['logit_batch']):
            logs={k:logits(v[off:off+spec['logit_batch']]+block.mlp.Down_bias) for k,v in states.items()};y=targets[off:off+spec['logit_batch']];ce={k:F.cross_entropy(v,y,reduction='none') for k,v in logs.items()};truth=logs['native'];baseline=logs['baseline'];effect=truth-baseline;ceeffect=ce['native']-ce['baseline']
            vals={'energy':effect.square().sum(-1),'ce_energy':ceeffect.square(),'native_ce':ce['native'],'baseline_ce':ce['baseline']}
            for m in spec['programs']:
                key=f'm{m}';log=logs[key];vals[key+'_error']=(log-truth).square().sum(-1);vals[key+'_ce_error']=(ce[key]-ce['native']).square();vals[key+'_ce_delta']=ce[key]-ce['native'];vals[key+'_top1_agreement']=(log.argmax(-1)==truth.argmax(-1)).float();vals[key+'_native_kl']=(truth.softmax(-1)*(truth.log_softmax(-1)-log.log_softmax(-1))).sum(-1)
            for name,v in vals.items():metrics.setdefault(name,[]);metrics[name].append(v)
        for name,vs in metrics.items():accumulate(kind+'_'+name,torch.cat(vs))
        counts[kind]=counts.get(kind,0)+len(pre)
    measure_states({'native':pre+write,'baseline':pre+meanwrite,**{f'm{m}':pre+p for m,p in pred.items()}},'intact')
    for h in range(9):
        changed=pre-hw[:,h];z=F.rms_norm(changed,(1152,));gold=B(z,z,L,R,D);delta={m:evaluate(m,z)-pred[m] for m in programs}
        measure_states({'native':changed+gold,'baseline':changed+write,**{f'm{m}':changed+write+d for m,d in delta.items()}},'mediation');print('head',h,'complete',flush=True)
    rng=np.random.default_rng(20261117);ix=rng.integers(cfg['rows'],size=(2000,cfg['rows']));results={}
    for kind in ['intact','mediation']:
        results[kind]={'native_ce':float(rows[kind+'_native_ce'].sum()/counts[kind]),'baseline_ce':float(rows[kind+'_baseline_ce'].sum()/counts[kind])}
        for m in spec['programs']:
            prefix=f'{kind}_m{m}';energy=rows[kind+'_energy'];e=rows[prefix+'_error'];ceenergy=rows[kind+'_ce_energy'];ceerror=rows[prefix+'_ce_error'];delta=rows[prefix+'_ce_delta'];perdoc=counts[kind]/cfg['rows']
            results[kind][f'm{m}']={'logit_fidelity':float(1-e.sum()/energy.sum()),'logit_fidelity_document_bootstrap95':np.quantile(1-e[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist(),'ce_effect_fidelity':float(1-ceerror.sum()/ceenergy.sum()),'ce_effect_document_bootstrap95':np.quantile(1-ceerror[ix].sum(1)/ceenergy[ix].sum(1),[.025,.975]).tolist(),'mean_ce_delta':float(delta.sum()/counts[kind]),'ce_delta_document_bootstrap95':np.quantile(delta[ix].sum(1)/counts[kind],[.025,.975]).tolist(),'native_top1_agreement':float(rows[prefix+'_top1_agreement'].sum()/counts[kind]),'mean_native_kl':float(rows[prefix+'_native_kl'].sum()/counts[kind])}
    gates={} if factorial else {'intact_replacement':results['intact']['m16']['mean_ce_delta']<=.02 and results['intact']['m16']['native_top1_agreement']>=.98,'conditional_mediation':results['mediation']['m16']['logit_fidelity']>=.9 and results['mediation']['m16']['ce_effect_fidelity']>=.8}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':spec,'results':results,'gates':gates,'counts':counts},indent=2));print(json.dumps({'results':results,'gates':gates},indent=2))
if __name__=='__main__':main()
