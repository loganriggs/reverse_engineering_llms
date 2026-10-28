#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bilin18_compiled_program import load_program,forward_normalized
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_context_role_compression import B
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_residual_replication'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);spec=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());cfg=json.loads((ART/'bilin18_l17_compiled_partners.json').read_text())['parent_config'];cfg={**cfg,'rows':spec['rows'],'corpus':spec['corpus'],'layer':17};model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,_,_=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);pred=forward_normalized(x,load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda'));error=native-pred;labels=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda();frozen=np.load(ART/'bilin18_l17_residual_scaling_frozen.npz');corrections={'compiled':torch.zeros_like(error),'full_oracle':error}
    for name in spec['maps']:
        encoder=torch.from_numpy(frozen[name+'_encoder']).cuda();decoder=torch.from_numpy(frozen[name+'_decoder']).cuda();corrections[name]=(error@encoder.T)@decoder.T
    def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
    truth=torch.cat([logits(pre[i:i+16]+native[i:i+16]+block.mlp.Down_bias) for i in range(0,len(pre),16)]);ce_native=F.cross_entropy(truth,labels,reduction='none');top=truth.topk(2,dim=-1).indices;rows={};results={}
    for name,c in corrections.items():
        accum={}
        for i in range(0,len(pre),16):
            sl=slice(i,i+16);q=logits(pre[sl]+pred[sl]+c[sl]+block.mlp.Down_bias);p=truth[sl];margin=q.gather(1,top[sl,:1]).squeeze(1)-q.gather(1,top[sl,1:]).squeeze(1);native_margin=p.gather(1,top[sl,:1]).squeeze(1)-p.gather(1,top[sl,1:]).squeeze(1)
            vals={'kl':(p.softmax(-1)*(p.log_softmax(-1)-q.log_softmax(-1))).sum(-1),'ce_delta':F.cross_entropy(q,labels[sl],reduction='none')-ce_native[sl],'top1':(p.argmax(-1)==q.argmax(-1)).float(),'margin_error':(margin-native_margin).square(),'logit_error':(q-p).square().sum(-1)}
            for k,v in vals.items():accum.setdefault(k,[]);accum[k].append(v)
        for k,v in accum.items():rows[name+'_'+k]=torch.cat(v).reshape(spec['rows'],-1).sum(1).cpu().numpy()
        results[name]={'mean_native_kl':float(rows[name+'_kl'].sum()/len(pre)),'mean_ce_delta':float(rows[name+'_ce_delta'].sum()/len(pre)),'native_top1_agreement':float(rows[name+'_top1'].sum()/len(pre)),'residual_energy_recovery':float(1-(error-c).square().sum()/error.square().sum())};print(name,results[name],flush=True)
    ix=np.random.default_rng(spec['seed']).integers(spec['rows'],size=(2000,spec['rows']));base=results['compiled'];difference=rows['fisher64_r16_kl']-rows['energy16_kl'];contrast={'value':float(difference.sum()/len(pre)),'paired_document_bootstrap95':np.quantile(difference[ix].sum(1)/len(pre),[.025,.975]).tolist()}
    for n,v in results.items():
        v['kl_recovery']=1-v['mean_native_kl']/base['mean_native_kl'];v['ce_delta_document_bootstrap95']=np.quantile(rows[n+'_ce_delta'][ix].sum(1)/len(pre),[.025,.975]).tolist();v['logit_error_recovery']=float(1-rows[n+'_logit_error'].sum()/rows['compiled_logit_error'].sum());v['margin_error_recovery']=float(1-rows[n+'_margin_error'].sum()/rows['compiled_margin_error'].sum())
    gates={'small_residual_repairs':any(results[n]['kl_recovery']>=.8 and results[n]['mean_ce_delta']<=.02 and results[n]['native_top1_agreement']>=.98 for n in ['energy16','fisher64_r16']),'fisher_directional_advantage':contrast['paired_document_bootstrap95'][1]<0,'broad_restoration_still_incomplete':all(results[n]['native_top1_agreement']<.98 for n in ['energy256','fisher64_r256'])}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':spec,'gates':gates,'results':results,'contrast':contrast,'native_ce':float(ce_native.mean())},indent=2));print('gates',gates)
if __name__=='__main__':main()
