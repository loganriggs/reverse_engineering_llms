"""H283 frozen mixture and shrinkage controls, fresh document evaluation."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from bilin18_residual_cross_program import intact_forward
from run_bilin18_l17_fixed_readout import read_features
from bilin18_readout_feature_families import features as family_features
from run_bilin18_l17_kl_intercept import distillation_kl
STEM=ART/'bilin18_l17_feature_mixture'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');pr=load_program(ART/'bilin18_l17_residual_cross_actual_refined_program.npz','cuda');ws={};maps={};selections={}
    for n,stem in cfg['stems'].items():
        f=load_program(ART/f'{stem}_frozen.npz','cuda')
        maps[n]=f
        for kind in ['mse','kl']:ws[f'{n}_{kind}']=f[kind]
        selections[n]=json.loads((ART/f'{stem}_selection.json').read_text())
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,_,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);base=compiled+intact_forward(x,pr);pred={'compiled':compiled,'constant':base,'native':native}
    corrections={}
    for n,f in maps.items():
        feat=read_features(x,pr,f['feature_mean'],f['feature_std']) if n=='linear' else family_features(x,pr,f)
        corrections[n]=feat@ws[n+'_kl'];pred[n+'_kl']=base+corrections[n]
    selection=json.loads((ART/'bilin18_l17_feature_mixture_selection.json').read_text());families=list(maps)
    for name in ['mixture','shrink','single']:
        weights=selection[name+'_weights'];pred[name]=base+sum(w*corrections[n] for w,n in zip(weights,families))
    labels=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda();rows={}
    def save(k,v):rows[k]=v.reshape(128,4).sum(1).cpu().numpy()
    save('residual_energy',(native-compiled).square().sum(-1))
    for n,v in pred.items():save(n+'_error',(v-native).square().sum(-1))
    def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
    metrics={n:{} for n in pred}
    for i in range(0,len(pre),16):
        sl=slice(i,i+16);truth=logits(pre[sl]+native[sl]+block.mlp.Down_bias);lp=truth.double().log_softmax(-1);ce=F.cross_entropy(truth.double(),labels[sl],reduction='none')
        for n,v in pred.items():
            q=logits(pre[sl]+v[sl]+block.mlp.Down_bias);vals={'kl':distillation_kl(q,lp),'ce_increase':F.cross_entropy(q.double(),labels[sl],reduction='none')-ce,'top1':(q.argmax(-1)==truth.argmax(-1)).float()}
            for k,val in vals.items():metrics[n].setdefault(k,[]);metrics[n][k].append(val)
    for n,vs in metrics.items():
        for k,v in vs.items():save(n+'_'+k,torch.cat(v))
    ix=np.random.default_rng(cfg['seed']).integers(128,size=(4000,128));results={};contrasts={};energy=rows['residual_energy'];klbase=rows['compiled_kl']
    for n in pred:
        kl=rows[n+'_kl'];ce=rows[n+'_ce_increase'];top=rows[n+'_top1'];results[n]={'residual_fidelity':float(1-rows[n+'_error'].sum()/energy.sum()),'mean_kl':float(kl.sum()/512),'kl_recovery':float(1-kl.sum()/klbase.sum()),'mean_kl_bootstrap95':np.quantile(kl[ix].sum(1)/512,[.025,.975]).tolist(),'mean_ce_increase':float(ce.sum()/512),'top1_agreement':float(top.sum()/512)}
    # Oracle chooses a single family separately for each fresh document using native KL.
    oracle=np.stack([rows[n+'_kl_kl'] for n in families]).min(0);rows['oracle_document_kl']=oracle
    pairs={'mixture_gain':('mixture','single'),'complementarity':('mixture','shrink')}
    for name,(n,b) in pairs.items():
        baseline=rows[b+'_kl'];d=baseline-rows[n+'_kl'];contrasts[name]={'relative_kl_reduction':float(d.sum()/baseline.sum()),'paired95':np.quantile(d[ix].sum(1)/baseline[ix].sum(1),[.025,.975]).tolist()}
    gates={name:c['relative_kl_reduction']>=.05 and c['paired95'][0]>0 for name,c in contrasts.items()};v=results['mixture'];gates['repair']=v['kl_recovery']>=.8 and v['mean_ce_increase']<=.02 and v['top1_agreement']>=.98;np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'contrasts':contrasts,'gates':gates,'training_audits':{n:s['audits'] for n,s in selections.items()},'selected':{n:{'mse':s['mse_winner'],'kl':s['kl_winner'],'checkpoint':s['selected'][s['kl_winner']]['step']} for n,s in selections.items()},'mixture_selection':selection,'oracle_document_mean_kl':float(oracle.sum()/512),'head_replay':replay},indent=2));print('GATES',gates,'RESULTS',results,flush=True)
if __name__=='__main__':main()
