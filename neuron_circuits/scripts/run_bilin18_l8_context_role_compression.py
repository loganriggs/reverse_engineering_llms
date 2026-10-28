#!/usr/bin/env python3
"""Compress shared context role; distinguish response rules from state functions."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_intervention_input_metric import roots

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_context_role_compression'

def B(a,b,L,R,D):
    return .5*((a@L.T)*(b@R.T)+(b@L.T)*(a@R.T))@D.T

def operator_metric(L,R,D):
    gram=D.T@D
    result=R.T@(gram*(L@L.T))@R+L.T@(gram*(R@R.T))@L
    cross=R.T@(gram*(L@R.T))@L
    return (result+cross+cross.T)

def projected(x,E,A):return (x@E.T)@A.T

def response_rules(x,y,mean,E,A,L,R,D):
    delta=y-x;m=(x+y)/2-mean;pm=projected(m,E,A);pd=projected(delta,E,A)
    linear=2*B(mean[None],delta,L,R,D)
    u=B(pm,delta,L,R,D);v=B(m,pd,L,R,D);w=B(pm,pd,L,R,D)
    return {'context_only':linear+2*u,'symmetric':linear+u+v,'both':linear+2*w,'cross':linear+2*u+2*v-2*w}

def state_rules(x,mean,E,A,L,R,D):
    y=x-mean;py=projected(y,E,A);linear=2*B(mean[None],y,L,R,D)
    cross=B(py,y,L,R,D);both=B(py,py,L,R,D)
    return {'symmetric':linear+cross,'both':linear+both,'cross':linear+2*cross-both}

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());torch.manual_seed(cfg['seed'])
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[8]
    L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
    frozen=np.load(STEM.parent/'bilin18_l8_intervention_input_metric_frozen.npz');old=torch.from_numpy(frozen['old_inputs']).cuda().float();mean=old.mean(0);z=old-mean
    C=(z.double().T@z.double()/len(z)).cpu();s,inv=roots(C)
    G=operator_metric(L.double(),R.double(),D.double()).cpu();G=(G+G.T)/2
    metric_error=0.
    for i in range(3):
        v=torch.randn(1152,device='cuda',dtype=torch.double)
        J=D.double()@((R.double()@v)[:,None]*L.double()+(L.double()@v)[:,None]*R.double())
        exact=float(J.square().sum());analytic=float(v.cpu()@G@v.cpu());metric_error=max(metric_error,abs(exact-analytic)/exact)
    _,pc=torch.linalg.eigh(C);pc=pc.flip(1)
    _,op=torch.linalg.eigh(s@G@s);op=op.flip(1)
    random=torch.linalg.qr(torch.randn(1152,1152,dtype=torch.double)).Q
    maps={};arrays={'mean':mean.cpu().numpy(),'operator_metric':G.numpy()};fit_quality={}
    for family in cfg['encoders']:
        for rank in cfg['ranks']:
            if family!='pca' and rank in [0,1152]:continue
            if family=='operator':E=op[:,:rank].T@inv;A=s@op[:,:rank]
            else:
                basis=(pc if family=='pca' else random)[:,:rank];E=basis.T;A=basis
            E=E.cuda().float();A=A.cuda().float();name=f'{family}{rank}';maps[name]=(E,A)
            arrays[name+'_encoder']=E.cpu().numpy();arrays[name+'_decoder']=A.cpu().numpy()
            error=(z-projected(z,E,A)).double().cpu()
            context=z.double().cpu()
            fit_quality[name]={'context_r2':float(1-error.square().sum()/context.square().sum()),
                               'operator_metric_r2':float(1-((error@G)*error).sum()/((context@G)*context).sum())}
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**arrays)
    print('All context encoders FROZEN; operator identity error',metric_error,flush=True)
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');assert cfg['start']+cfg['rows']<=len(corpus)
    pre,hw,replay=collect_sources(model,corpus,cfg);dim=pre.shape[-1];x=F.rms_norm(pre,(dim,));native=B(x,x,L,R,D);old_native=B(old,old,L,R,D);old_mean=old_native.mean(0)
    changed={f'h{h}_s{scale:g}':F.rms_norm(pre-scale*hw[:,h],(dim,)) for h in range(9) for scale in cfg['scales']}
    targets={key:B(y,y,L,R,D)-native for key,y in changed.items()}
    np.savez_compressed(STEM.with_name(STEM.name+'_test_inputs.npz'),native=x.cpu().numpy(),**{k:y.cpu().numpy() for k,y in changed.items()})
    rowarrays={};results={};cycles={};natural={};head_results={};full_error=0.;full_energy=0.;state_diff_error=0.;state_diff_energy=0.
    for scale in cfg['scales']:
        key=f'{scale:g}';rowarrays[key+'_energy']=sum(targets[f'h{h}_s{key}'].square().sum(-1).reshape(cfg['rows'],-1).sum(1).cpu().numpy() for h in range(9))
    for name,(E,A) in maps.items():
        old_states=state_rules(old,mean,E,A,L,R,D);states=state_rules(x,mean,E,A,L,R,D)
        natural[name]={}
        for rule,pred in states.items():
            prediction=pred+old_mean-old_states[rule].mean(0)
            natural[name][rule]=float(1-(prediction-native).square().sum()/(native-old_mean).square().sum())
        for scale in cfg['scales']:
            key=f'{scale:g}';energy=rowarrays[key+'_energy'];accum={r:np.zeros(cfg['rows']) for r in cfg['rules']}
            for h in range(9):
                y=changed[f'h{h}_s{key}'];target=targets[f'h{h}_s{key}'];predictions=response_rules(x,y,mean,E,A,L,R,D)
                for rule,pred in predictions.items():
                    error=(pred-target).square().sum(-1);accum[rule]+=error.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                    head_results[f'{name}_{rule}_h{h}_s{key}']=float(1-error.sum()/target.square().sum())
                    if name=='pca1152':full_error+=float(error.sum());full_energy+=float(target.square().sum())
                if h==1 and scale==1:
                    changed_states=state_rules(y,mean,E,A,L,R,D)
                    for rule in states:
                        state_diff_error+=float((changed_states[rule]-states[rule]-predictions[rule]).square().sum());state_diff_energy+=float(predictions[rule].square().sum())
            results[f'{name}_s{key}']={r:float(1-err.sum()/energy.sum()) for r,err in accum.items()}
            for rule,err in accum.items():rowarrays[f'{name}_{rule}_s{key}_error']=err
        y=changed['h1_s1'];zz=changed['h3_s1']
        xy=response_rules(x,y,mean,E,A,L,R,D);yz=response_rules(y,zz,mean,E,A,L,R,D);zx=response_rules(zz,x,mean,E,A,L,R,D)
        denom=targets['h1_s1'].square().sum()
        cycles[name]={r:float(((xy[r]+yz[r]+zx[r]).square().sum()/denom).sqrt()) for r in cfg['rules']}
        print(name,'full',results[name+'_s1'],'natural',natural[name],'cycle',cycles[name],flush=True)
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rowarrays)
    ix=np.random.default_rng(cfg['seed']).integers(cfg['rows'],size=(2000,cfg['rows']));intervals={}
    for name in maps:
        for rule in cfg['rules']:
            error=rowarrays[f'{name}_{rule}_s1_error'];energy=rowarrays['1_energy']
            intervals[f'{name}_{rule}']=np.quantile(1-error[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()
    candidates=[n for n in maps if int(''.join(filter(str.isdigit,n)))<=64]
    full_relative=(full_error/full_energy)**.5;state_relative=(state_diff_error/state_diff_energy)**.5
    gates={'compact_response':any(results[n+'_s1']['context_only']>=.8 for n in candidates),
           'compact_integrable':any(results[n+'_s1'][r]>=.8 and natural[n][r]>=.8 for n in candidates for r in ['symmetric','both','cross']),
           'operator_advantage':results['operator64_s1']['context_only']-results['pca64_s1']['context_only']>=.1,
           'consistency':full_relative<=1e-4 and state_relative<=1e-4 and max(c[r] for c in cycles.values() for r in ['symmetric','both','cross'])<=1e-4}
    output={'config':cfg,'gates':gates,'head_replay':replay,'operator_metric_relative_error':metric_error,'full_rank_replay_relative_rms':full_relative,'state_difference_relative_rms':state_relative,'discovery_fit':fit_quality,'natural_r2':natural,'responses':results,'cycles_relative_rms':cycles,'head_results':head_results,'full_response_95':intervals}
    STEM.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=['# Shared context-role compression of full MLP response','','No per-state SVD or output-axis restriction. Context-only rules need not integrate to a state function.','','| Context code | Context-only response | Symmetric response | Both-role response | Cross response | Symmetric natural R2 | Cross natural R2 | Context-only cycle RMS |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in maps:
        r=results[name+'_s1'];n=natural[name]
        lines.append(f"| {name} | {r['context_only']:.3f} | {r['symmetric']:.3f} | {r['both']:.3f} | {r['cross']:.3f} | {n['symmetric']:.3f} | {n['cross']:.3f} | {cycles[name]['context_only']:.3f} |")
    lines+=['',f'Locked gates: {gates}. Full-rank replay relative RMS {full_relative:.3g}; state-difference identity {state_relative:.3g}; operator metric identity {metric_error:.3g}.','',cfg['cost']]
    STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n');print(gates,flush=True)

if __name__=='__main__':main()
