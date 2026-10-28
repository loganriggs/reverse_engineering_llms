"""H261 fixed heldout-head comparisons plus natural-state tradeoff audit."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_residual_feature_objective import raw_features
STEM=ART/'bilin18_l17_response_supervision'
@torch.inference_mode()
def main():
    j=json.loads(STEM.with_suffix('.json').read_text());cfg=j['config'];rows=np.load(STEM.with_name(STEM.name+'_rows.npz'));ix=np.random.default_rng(cfg['seed']+2).integers(128,size=(2000,128));summary={};contrasts={}
    for payload in cfg['payloads']:
        for scale in cfg['scales']:
            for group,hs in [('supervised',cfg['supervised_heads']),('heldout',cfg['heldout_heads'])]:
                key=f'{payload}_s{scale}_{group}'
                def total(m):return sum(rows[f'{payload}_s{scale}_h{h}_{m}'] for h in hs)
                energy=total('residual_energy');summary[key]={}
                for name in cfg['models']:
                    e=total(name+'_residual_error');kl=total(name+'_kl');summary[key][name]={'residual_fidelity':float(1-e.sum()/energy.sum()),'mean_conditional_kl':float(kl.sum()/(128*4*len(hs))),'residual_fidelity_document_bootstrap95':np.quantile(1-e[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
                contrasts[key]={}
                for family in cfg['families']:
                    base=family+'_beta0';e0=total(base+'_residual_error');k0=total(base+'_kl')
                    for beta in [1,10]:
                        name=f'{family}_beta{beta}';de=e0-total(name+'_residual_error');dk=k0-total(name+'_kl');contrasts[key][name]={'residual_gain':float(de.sum()/energy.sum()),'residual_gain_bootstrap95':np.quantile(de[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist(),'relative_kl_gain':float(dk.sum()/k0.sum()),'kl_gain_bootstrap95':np.quantile(dk[ix].sum(1)/k0[ix].sum(1),[.025,.975]).tolist()}
    def gate(group,metric,ci):return all(contrasts[f'actual_s{s}_{group}']['quadratic1152_beta1'][metric]>=.1 and contrasts[f'actual_s{s}_{group}']['quadratic1152_beta1'][ci][0]>0 for s in cfg['scales'])
    j['groups']=summary;j['group_contrasts']=contrasts;j['gates']={'heldout_transfer':gate('heldout','residual_gain','residual_gain_bootstrap95'),'supervised_learning':gate('supervised','residual_gain','residual_gain_bootstrap95'),'behavioral_transfer':gate('heldout','relative_kl_gain','kl_gain_bootstrap95')}
    # Natural-state cost: same frozen fits, no model selection.
    torch.set_num_threads(6);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');frozen=load_program(STEM.with_name(STEM.name+'_frozen.npz'),'cuda');selection=json.loads(STEM.with_name(STEM.name+'_selection.json').read_text())['selected'];corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,_,_=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);error=native-compiled;labels=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda();corrections={'compiled':torch.zeros_like(x)}
    for family in cfg['families']:
        f=(raw_features(x,frozen['basis'],frozen['input_mean'],family)-frozen[family+'_mean'])/frozen[family+'_std'];f=torch.cat([f,torch.ones(len(f),1,device='cuda')],1)
        for beta in cfg['response_weights']:
            name=f'{family}_beta{beta}';corrections[name]=f@frozen[selection[name]]
    def logits(s):return 30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)
    natural={}
    for name,c in corrections.items():
        vs=[]
        for i in range(0,len(x),16):
            sl=slice(i,i+16);t=logits(pre[sl]+native[sl]+block.mlp.Down_bias);q=logits(pre[sl]+compiled[sl]+c[sl]+block.mlp.Down_bias);kl=(t.double().softmax(-1)*(t.double().log_softmax(-1)-q.double().log_softmax(-1))).sum(-1);ce=F.cross_entropy(q,labels[sl],reduction='none')-F.cross_entropy(t,labels[sl],reduction='none');top=(q.argmax(-1)==t.argmax(-1)).double();vs.append(torch.stack([kl,ce,top],1))
        v=torch.cat(vs).mean(0);natural[name]={'kl':float(v[0]),'ce_delta':float(v[1]),'top1':float(v[2]),'residual_energy_recovery':float(1-(error-c).square().sum()/error.square().sum())}
    for v in natural.values():v['kl_recovery']=1-v['kl']/natural['compiled']['kl']
    j['natural']=natural;STEM.with_suffix('.json').write_text(json.dumps(j,indent=2));print('GATES',j['gates'],'NATURAL',natural)
    for k in ['actual_s0.1_supervised','actual_s0.1_heldout','actual_s1.0_supervised','actual_s1.0_heldout']:print(k,summary[k],contrasts[k])
if __name__=='__main__':main()
