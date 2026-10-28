#!/usr/bin/env python3
"""Compiled-state-selected local downstream charts with oracle residual coefficients."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bilin18_compiled_program import load_program,forward_normalized
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_role_compression import B
from run_bilin18_l17_residual_restoration import score_gradients
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_local_residual_charts'

def local_chart(state,weight,candidates,rank):
    logits=30*torch.tanh((F.rms_norm(state,(state.shape[-1],))@weight.T)/30);p=logits.softmax(-1);ids=p.topk(candidates,dim=-1).indices;grad,_=score_gradients(state,weight,ids);weighted=grad*p.gather(1,ids).sqrt()[:,:,None]
    _,s,vh=torch.linalg.svd(weighted,full_matrices=False);chart=torch.linalg.qr(vh[:,:rank].transpose(1,2)).Q.transpose(1,2)
    return chart,s,p.gather(1,ids).sum(-1),ids

def local_project(error,chart):return torch.einsum('nr,nrd->nd',torch.einsum('nd,nrd->nr',error,chart),chart)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,_,_=collect_sources(model,corpus,{**cfg,'layer':17});x=F.rms_norm(pre,(1152,));program=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');pred=forward_normalized(x,program);state=pre+pred+block.mlp.Down_bias;charts=[];spectra=[];masses=[];ids=[]
    for off in range(0,len(pre),cfg['chart_batch']):
        c,s,p,i=local_chart(state[off:off+cfg['chart_batch']],model.lm_head.weight,cfg['candidate_tokens'],max(cfg['ranks']));charts.append(c);spectra.append(s);masses.append(p);ids.append(i)
    chart=torch.cat(charts);spectrum=torch.cat(spectra);mass=torch.cat(masses);ids=torch.cat(ids);ortho=float((chart@chart.transpose(1,2)-torch.eye(max(cfg['ranks']),device='cuda')).abs().max());np.savez_compressed(STEM.with_name(STEM.name+'_selection.npz'),singular_values=spectrum.cpu().numpy(),probability_mass=mass.cpu().numpy(),token_ids=ids.cpu().numpy());print('Compiled-state charts selected before native error calculation',flush=True)
    native=B(x,x,L,R,D);error=native-pred;wrong=(torch.arange(len(pre),device='cuda')+len(cfg['positions']))%len(pre);corrections={'compiled':torch.zeros_like(error),'full_oracle':error}
    for rank in cfg['ranks']:
        corrections[f'local{rank}']=local_project(error,chart[:,:rank]);corrections[f'wrong{rank}']=local_project(error,chart[wrong,:rank])
    frozen=np.load(ART/'bilin18_l17_residual_scaling_frozen.npz')
    for rank in [16,64]:
        for family in ['energy','fisher64_r']:
            name=f'{family}{rank}';e=torch.from_numpy(frozen[name+'_encoder']).cuda();a=torch.from_numpy(frozen[name+'_decoder']).cuda();corrections[f'global_{name}']=(error@e.T)@a.T
            if family=='fisher64_r':q=torch.linalg.qr(a).Q;corrections[f'global_fisher_orth{rank}']=(error@q)@q.T
    labels=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda()
    def logits(s):return 30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)
    truth=torch.cat([logits(pre[i:i+16]+native[i:i+16]+block.mlp.Down_bias) for i in range(0,len(pre),16)]);ce_native=F.cross_entropy(truth,labels,reduction='none');top=truth.topk(2,dim=-1).indices;rows={};results={}
    for name,c in corrections.items():
        accum={}
        for i in range(0,len(pre),cfg['logit_batch']):
            sl=slice(i,i+cfg['logit_batch']);q=logits(state[sl]+c[sl]);p=truth[sl];margin=q.gather(1,top[sl,:1]).squeeze(1)-q.gather(1,top[sl,1:]).squeeze(1);native_margin=p.gather(1,top[sl,:1]).squeeze(1)-p.gather(1,top[sl,1:]).squeeze(1)
            vals={'kl':(p.softmax(-1)*(p.log_softmax(-1)-q.log_softmax(-1))).sum(-1),'ce_delta':F.cross_entropy(q,labels[sl],reduction='none')-ce_native[sl],'top1':(p.argmax(-1)==q.argmax(-1)).float(),'margin_error':(margin-native_margin).square(),'logit_error':(q-p).square().sum(-1)}
            for k,v in vals.items():accum.setdefault(k,[]);accum[k].append(v)
        for k,v in accum.items():rows[name+'_'+k]=torch.cat(v).reshape(cfg['rows'],-1).sum(1).cpu().numpy()
        results[name]={'mean_native_kl':float(rows[name+'_kl'].sum()/len(pre)),'mean_ce_delta':float(rows[name+'_ce_delta'].sum()/len(pre)),'native_top1_agreement':float(rows[name+'_top1'].sum()/len(pre)),'residual_energy_recovery':float(1-(error-c).square().sum()/error.square().sum())};print(name,results[name],flush=True)
    ix=np.random.default_rng(cfg['seed']).integers(cfg['rows'],size=(2000,cfg['rows']));base=results['compiled'];contrasts={}
    for name,v in results.items():
        v['kl_recovery']=1-v['mean_native_kl']/base['mean_native_kl'];v['logit_error_recovery']=float(1-rows[name+'_logit_error'].sum()/rows['compiled_logit_error'].sum());v['margin_error_recovery']=float(1-rows[name+'_margin_error'].sum()/rows['compiled_margin_error'].sum());v['ce_delta_document_bootstrap95']=np.quantile(rows[name+'_ce_delta'][ix].sum(1)/len(pre),[.025,.975]).tolist()
    for control in ['wrong16','global_fisher64_r16','global_fisher_orth16']:
        d=rows['local16_kl']-rows[control+'_kl'];contrasts[control]={'local_minus_control_kl':float(d.sum()/len(pre)),'paired_document_bootstrap95':np.quantile(d[ix].sum(1)/len(pre),[.025,.975]).tolist()}
    local=results['local16'];gates={'local_small_repair':local['kl_recovery']>=.8 and local['mean_ce_delta']<=.02 and local['native_top1_agreement']>=.98,'context_specificity':local['mean_native_kl']<=.75*results['wrong16']['mean_native_kl'] and contrasts['wrong16']['paired_document_bootstrap95'][1]<0,'local_beats_global':all(local['mean_native_kl']<=.75*results[n]['mean_native_kl'] for n in ['global_fisher64_r16','global_fisher_orth16'])}
    selection={'probability_mass_mean':float(mass.mean()),'probability_mass_min':float(mass.min()),'probability_mass_quantiles':np.quantile(mass.cpu().numpy(),[.1,.5,.9]).tolist(),'orthogonality_error':ortho,'truncated_fisher_spectrum_fraction_mean':{str(r):float((spectrum[:,:r].square().sum(-1)/spectrum.square().sum(-1)).mean()) for r in cfg['ranks']}}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'gates':gates,'results':results,'contrasts':contrasts,'selection':selection},indent=2));print('gates',gates,flush=True)
if __name__=='__main__':main()
