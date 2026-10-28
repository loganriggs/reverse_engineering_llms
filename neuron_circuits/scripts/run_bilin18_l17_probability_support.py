"""H255 fixed probability support/entropy sweep of local oracle charts."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,local_chart,local_project,load_full_model,collect_sources,load_program,forward_normalized,B
STEM=ART/'bilin18_l17_probability_support'

def summarize(kl,base,mask,ix):
    n=int(mask.sum())
    if n==0:return {'sites':0}
    num=(kl*mask).reshape(len(ix[0]),-1).sum(1);den=(base*mask).reshape(len(ix[0]),-1).sum(1)
    ratios=1-num[ix].sum(1)/den[ix].sum(1)
    return {'sites':n,'mean_kl':float(num.sum()/n),'baseline_kl':float(den.sum()/n),'recovery':float(1-num.sum()/den.sum()),'document_bootstrap95':np.nanquantile(ratios,[.025,.975]).tolist()}

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17]
    L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
    pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),{**cfg,'layer':17});x=F.rms_norm(pre,(1152,));pred=forward_normalized(x,load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda'));state=pre+pred+block.mlp.Down_bias
    def logp(s):return (30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)).log_softmax(-1)
    compiled=torch.cat([logp(state[i:i+16]) for i in range(0,len(state),16)])
    entropy=(-(compiled.exp()*compiled).sum(-1)).cpu().numpy();data={'entropy':entropy};charts={};selection={}
    for support in cfg['supports']:
        cs=[];ms=[]
        for i in range(0,len(state),4):
            c,_,m,_=local_chart(state[i:i+4],model.lm_head.weight,support,64);cs.append(c);ms.append(m)
        charts[support]=torch.cat(cs);data[f'mass{support}']=torch.cat(ms).cpu().numpy();print('charts',support,flush=True)
    # All bases frozen before native errors are computed.
    error=B(x,x,L,R,D)-pred;truth=torch.cat([logp(state[i:i+16]+error[i:i+16]) for i in range(0,len(state),16)])
    data['baseline_kl']=(truth.exp()*(truth-compiled)).sum(-1).cpu().numpy()
    wrong=(torch.arange(len(state),device='cuda')+len(cfg['positions']))%len(state)
    for support,chart in charts.items():
        for rank in cfg['ranks']:
            for mode,q in [('local',chart),('wrong',chart[wrong])]:
                correction=local_project(error,q[:,:rank]);vals=[]
                for i in range(0,len(state),16):
                    p=truth[i:i+16];z=logp(state[i:i+16]+correction[i:i+16]);vals.append((p.exp()*(p-z)).sum(-1))
                data[f'{mode}_s{support}_r{rank}']=torch.cat(vals).cpu().numpy()
    ix=np.random.default_rng(cfg['seed']).integers(cfg['rows'],size=(2000,cfg['rows']));masks={'all':np.ones(len(state),dtype=bool),'low':entropy<2,'medium':(entropy>=2)&(entropy<4),'high':entropy>=4};results={}
    for name,mask in masks.items():
        results[name]={k:summarize(v,data['baseline_kl'],mask,ix) for k,v in data.items() if k.startswith(('local','wrong'))}
        selection[name]={'sites':int(mask.sum()),'mean_entropy':float(entropy[mask].mean()) if mask.any() else None,'probability_mass':{str(s):float(data[f'mass{s}'][mask].mean()) if mask.any() else None for s in cfg['supports']}}
    stable=all(max(results[b][f'local_s{s}_r16']['recovery'] for s in cfg['supports'])-min(results[b][f'local_s{s}_r16']['recovery'] for s in cfg['supports'])<=.05 for b in masks if selection[b]['sites']>=30)
    def boot(mask):
        num=(data['local_s1024_r16']*mask).reshape(cfg['rows'],-1).sum(1);den=(data['baseline_kl']*mask).reshape(cfg['rows'],-1).sum(1);return 1-num[ix].sum(1)/den[ix].sum(1)
    gap=results['low']['local_s1024_r16']['recovery']-results['high']['local_s1024_r16']['recovery'];ci=np.nanquantile(boot(masks['low'])-boot(masks['high']),[.025,.975]).tolist();contrast={'low_minus_high_recovery':gap,'document_bootstrap95':ci}
    gates={'support_stability':stable,'entropy_dependence':min(selection['low']['sites'],selection['high']['sites'])>=30 and gap>=.15 and ci[0]>0}
    np.savez_compressed(STEM.with_name(STEM.name+'_sites.npz'),**data);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'selection':selection,'results':results,'contrast':contrast,'gates':gates},indent=2));print(json.dumps({'selection':selection,'contrast':contrast,'gates':gates}),flush=True)
if __name__=='__main__':main()
