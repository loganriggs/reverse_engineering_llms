"""H254: distinguish residual organization from generic readout sensitivity."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,local_chart,local_project,load_full_model,collect_sources,load_program,forward_normalized,B
STEM=ART/'bilin18_l17_residual_geometry_controls'

def match_norm(direction,reference):
    return direction*(reference.norm(dim=-1,keepdim=True)/direction.norm(dim=-1,keepdim=True).clamp_min(1e-30))

@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
    block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
    corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r');pre,_,_=collect_sources(model,corpus,{**cfg,'layer':17})
    x=F.rms_norm(pre,(1152,));pred=forward_normalized(x,load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda'));state=pre+pred+block.mlp.Down_bias
    chart=torch.cat([local_chart(state[i:i+8],model.lm_head.weight,256,64)[0] for i in range(0,len(state),8)])
    error=B(x,x,L,R,D)-pred;npos=len(cfg['positions']);wrong=(torch.arange(len(state),device='cuda')+npos)%len(state)
    families={'native':error};norm_audit={}
    for seed in cfg['seeds']:
        gen=torch.Generator(device='cuda').manual_seed(seed)
        families[f'isotropic{seed}']=match_norm(torch.randn(error.shape,device='cuda',generator=gen),error)
        # A random nonzero cyclic document offset gives a derangement and preserves position.
        offset=int(np.random.default_rng(seed).integers(1,cfg['rows']))*npos
        families[f'shuffled{seed}']=match_norm(error.roll(offset,0),error)
    def logp(s):return (30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)).log_softmax(-1)
    entropy=torch.cat([-(logp(state[i:i+16]).exp()*logp(state[i:i+16])).sum(-1) for i in range(0,len(state),16)]).cpu().numpy()
    rows={};results={};sites={"entropy":entropy}
    for family,e in families.items():
        norm_audit[family]=float(((e.norm(dim=-1)-error.norm(dim=-1)).abs()/error.norm(dim=-1)).max())
        corrections={'none':torch.zeros_like(e),'full':e}
        for rank in cfg['ranks']:
            corrections[f'local{rank}']=local_project(e,chart[:,:rank]);corrections[f'wrong{rank}']=local_project(e,chart[wrong,:rank])
        for scale in cfg['scales']:
            group=f'{family}_s{scale}';results[group]={}
            for name,c in corrections.items():
                vals=[]
                for i in range(0,len(state),16):
                    sl=slice(i,i+16);p=logp(state[sl]+scale*e[sl]);q=logp(state[sl]+scale*c[sl]);vals.append((p.exp()*(p-q)).sum(-1))
                key=f'{group}_{name}';sites[key]=torch.cat(vals).cpu().numpy();rows[key]=sites[key].reshape(cfg['rows'],npos).sum(-1)
                results[group][name]={'kl':float(rows[key].mean()/npos)}
            base=results[group]['none']['kl']
            for v in results[group].values():v['recovery']=1-v['kl']/base
            print(group,results[group],flush=True)
    ix=np.random.default_rng(20261126).integers(cfg['rows'],size=(2000,cfg['rows']))
    summary={};contrasts={}
    for scale in cfg['scales']:
        summary[str(scale)]={};boot={}
        for family in ['native','isotropic','shuffled']:
            names=['native'] if family=='native' else [f'{family}{s}' for s in cfg['seeds']]
            def pooled(name):return np.mean([rows[f'{f}_s{scale}_{name}'] for f in names],axis=0)
            base=pooled('none');stats={}
            for name in ['local16','wrong16','local64','wrong64']:
                values=pooled(name);b=1-values[ix].sum(-1)/base[ix].sum(-1);boot[(family,name)]=b
                stats[name]={'recovery':float(1-values.sum()/base.sum()),'document_bootstrap95':np.quantile(b,[.025,.975]).tolist()}
            summary[str(scale)][family]=stats
        contrasts[str(scale)]={}
        for control in ['isotropic','shuffled']:
            d=boot[('native','local16')]-boot[(control,'local16')]
            contrasts[str(scale)][control]={'native_minus_control':summary[str(scale)]['native']['local16']['recovery']-summary[str(scale)][control]['local16']['recovery'],'paired_document_bootstrap95':np.quantile(d,[.025,.975]).tolist()}
    gates={'generic_sufficiency':all(v['isotropic']['local16']['recovery']>=.6 and v['isotropic']['local16']['recovery']-v['isotropic']['wrong16']['recovery']>=.4 for v in summary.values()),'native_specialness':all(v['native_minus_control']>=.1 and v['paired_document_bootstrap95'][0]>0 for c in contrasts.values() for v in c.values())}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows)
    np.savez_compressed(STEM.with_name(STEM.name+'_sites.npz'),**sites)
    STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'summary':summary,'contrasts':contrasts,'gates':gates,'norm_relative_error_max':norm_audit},indent=2))
    print('SUMMARY',json.dumps(summary));print('CONTRASTS',json.dumps(contrasts));print('GATES',gates)
if __name__=='__main__':main()
