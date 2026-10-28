"""H281 frozen readout-supervision learning curve, fresh document evaluation."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from bilin18_residual_cross_program import intact_forward
from run_bilin18_l17_fixed_readout import read_features
from run_bilin18_l17_kl_intercept import distillation_kl
STEM=ART/'bilin18_l17_readout_learning'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');pr=load_program(ART/'bilin18_l17_residual_cross_actual_refined_program.npz','cuda');ws={};selections={}
    for n,stem in cfg['stems'].items():
        f=load_program(ART/f'{stem}_frozen.npz','cuda')
        if not ws:fm=f['feature_mean'];fs=f['feature_std']
        else:assert torch.equal(fm,f['feature_mean']) and torch.equal(fs,f['feature_std'])
        for kind in ['mse','kl']:ws[f'n{n}_{kind}']=f[kind]
        selections[n]=json.loads((ART/f'{stem}_selection.json').read_text())
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,_,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);base=compiled+intact_forward(x,pr);features=read_features(x,pr,fm,fs);pred={'compiled':compiled,'constant':base,'native':native,**{n:base+features@w for n,w in ws.items()}};labels=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda();rows={}
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
    pairs={f'{kind}_{lo}_{hi}':(f'n{hi}_{kind}',f'n{lo}_{kind}') for kind in ['mse','kl'] for lo,hi in [(192,384),(384,768),(192,768)]};pairs['objective_gain']=('n768_kl','n768_mse')
    for name,(n,b) in pairs.items():
        baseline=rows[b+'_kl'];d=baseline-rows[n+'_kl'];contrasts[name]={'relative_kl_reduction':float(d.sum()/baseline.sum()),'paired95':np.quantile(d[ix].sum(1)/baseline[ix].sum(1),[.025,.975]).tolist()}
    gates={name:contrasts[key]['relative_kl_reduction']>=threshold and contrasts[key]['paired95'][0]>0 for name,key,threshold in [('learning_kl','kl_192_768',.05),('learning_mse','mse_192_768',.05),('objective_gain','objective_gain',.1)]};v=results['n768_kl'];gates['repair']=v['kl_recovery']>=.8 and v['mean_ce_increase']<=.02 and v['top1_agreement']>=.98;np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'contrasts':contrasts,'gates':gates,'training_audits':{n:s['audits'] for n,s in selections.items()},'selected':{n:{'mse':s['mse_winner'],'kl':s['kl_winner'],'checkpoint':s['selected'][s['kl_winner']]['step']} for n,s in selections.items()},'head_replay':replay},indent=2));print('GATES',gates,'RESULTS',results,flush=True)
if __name__=='__main__':main()
