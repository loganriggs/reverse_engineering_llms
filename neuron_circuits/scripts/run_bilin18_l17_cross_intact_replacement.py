"""H276 intact final-MLP substitution, no conditional native write supplied."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from bilin18_residual_cross_program import forward,intact_forward,counts
STEM=ART/'bilin18_l17_cross_intact_replacement'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');ps={n:load_program(ART/f'bilin18_l17_residual_cross_{n}_program.npz','cuda') for n in ['actual','mixture']};mu=ps['actual']['mean'];pd={k:v.double() for k,v in p.items()};anchor64=B(mu[None].double(),mu[None].double(),L.double(),R.double(),D.double())-forward_normalized(mu[None].double(),pd);anchor=anchor64[0].float();costs={}
    for n,pr in ps.items():
        pr['anchor']=anchor;path=ART/f'bilin18_l17_residual_cross_{n}_intact_program.npz';np.savez_compressed(path,**{k:v.cpu().numpy() for k,v in pr.items()});costs[n]={'correction_scalars':sum(counts(pr).values()),'compiler_scalars':sum(v.numel() for v in p.values()),'saved_bytes':path.stat().st_size}
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,_,replay=collect_sources(model,corpus,cfg);x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);pred={'compiled':compiled,'constant':compiled+anchor,'common':compiled+anchor+forward(x,ps['actual'],False),'actual':compiled+intact_forward(x,ps['actual']),'mixture':compiled+intact_forward(x,ps['mixture']),'native':native};rows={};labels=torch.from_numpy(np.array(corpus[:,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda()
    def save(k,v):rows[k]=v.reshape(128,4).sum(1).cpu().numpy()
    save('residual_energy',(native-compiled).square().sum(-1));save('native_energy',native.square().sum(-1))
    for n,v in pred.items():save(n+'_error',(v-native).square().sum(-1))
    def logits(v):return 30*torch.tanh(model.lm_head(F.rms_norm(v,(1152,)))/30)
    metrics={n:{} for n in pred}
    for i in range(0,len(pre),16):
        sl=slice(i,i+16);truth=logits(pre[sl]+native[sl]+block.mlp.Down_bias);lp=truth.double().log_softmax(-1);prob=lp.exp();ce=F.cross_entropy(truth.double(),labels[sl],reduction='none')
        for n,v in pred.items():
            q=logits(pre[sl]+v[sl]+block.mlp.Down_bias);lq=q.double().log_softmax(-1);vals={'kl':(prob*(lp-lq)).sum(-1),'ce_increase':F.cross_entropy(q.double(),labels[sl],reduction='none')-ce,'top1':(q.argmax(-1)==truth.argmax(-1)).float(),'label_accuracy':(q.argmax(-1)==labels[sl]).float()}
            for k,val in vals.items():metrics[n].setdefault(k,[]);metrics[n][k].append(val)
    for n,vs in metrics.items():
        for k,v in vs.items():save(n+'_'+k,torch.cat(v))
    ix=np.random.default_rng(20261224).integers(128,size=(4000,128));results={};contrasts={};energy=rows['residual_energy'];klbase=rows['compiled_kl']
    for n in cfg['models']:
        kl=rows[n+'_kl'];ce=rows[n+'_ce_increase'];top=rows[n+'_top1'];results[n]={'residual_fidelity':float(1-rows[n+'_error'].sum()/energy.sum()),'native_write_fidelity':float(1-rows[n+'_error'].sum()/rows['native_energy'].sum()),'mean_kl':float(kl.sum()/512),'kl_recovery':float(1-kl.sum()/klbase.sum()),'kl_recovery_bootstrap95':np.quantile(1-kl[ix].sum(1)/klbase[ix].sum(1),[.025,.975]).tolist(),'mean_ce_increase':float(ce.sum()/512),'ce_increase_bootstrap95':np.quantile(ce[ix].sum(1)/512,[.025,.975]).tolist(),'top1_agreement':float(top.sum()/512),'top1_bootstrap95':np.quantile(top[ix].sum(1)/512,[.025,.975]).tolist(),'next_token_accuracy':float(rows[n+'_label_accuracy'].sum()/512)}
    for n in ['actual','mixture']:
        for b in ['constant','common']:
            d=rows[b+'_kl']-rows[n+'_kl'];contrasts[n+'_vs_'+b]={'relative_kl_reduction':float(d.sum()/rows[b+'_kl'].sum()),'paired95':np.quantile(d[ix].sum(1)/rows[b+'_kl'][ix].sum(1),[.025,.975]).tolist()}
    gates={}
    for n in ['actual','mixture']:gates['intact_'+n]=results[n]['mean_ce_increase']<=.02 and results[n]['top1_agreement']>=.98;gates['repair_'+n]=gates['intact_'+n] and results[n]['kl_recovery']>=.8
    audit={'anchor_float32_relative_error':float((anchor.double()-anchor64[0]).norm()/anchor64.norm()),'anchor_norm':float(anchor.norm()),'common_at_mean_norm':float(forward(mu[None],ps['actual']).norm())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'contrasts':contrasts,'gates':gates,'costs':costs,'audits':audit,'head_replay':replay},indent=2));print('GATES',gates,'RESULTS',results,flush=True)
if __name__=='__main__':main()
