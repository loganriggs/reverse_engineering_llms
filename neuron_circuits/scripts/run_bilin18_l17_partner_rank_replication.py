"""H273 frozen partner-rank replication and matched-angle transfer."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l17_residual_term_accounting import mean_linear
from run_bilin18_l17_direct_partner_fit import partner_q
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_partner_rank_replication'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight.double() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda',torch.float64);qs={};models={}
    for rank,stem in cfg['stems'].items():
        f=load_program(ART/f'{stem}_frozen.npz','cuda',torch.float64);cc=torch.linalg.qr(f['core']).Q
        if not qs:c=cc;mu=f['mean']
        else:assert float((cc@cc.T-c@c.T).abs().max())<1e-6 and torch.equal(f['mean'],mu)
        for n in ['weighted','optimized']:qs[f'k{rank}_{n}']=partner_q(f[n],c)
        models[rank]=json.loads((ART/f'{stem}_selection.json').read_text())['winner']
    pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double();err=lambda z:B(z,z,L,R,D)-forward_normalized(z,p);e0=err(x);rows={};gen=torch.Generator(device='cuda').manual_seed(cfg['seed'])
    for h in cfg['heads']:
        actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
        for label,pay in [('actual',actual),('angle',angle)]:
            for scale in cfg['scales']:
                z=F.rms_norm(pre-scale*pay,(1152,)).double();target=err(z)-e0;t=quadratic_terms(x,z,mu,c,L,R,D,p);lin=mean_linear(z-x,mu,L,R,D,p);prefix=f'{label}_s{scale}_h{h}'
                for metric,v in [('cross_energy',t[:,1].square().sum(-1)),('total_energy',target.square().sum(-1)),('all_total',(lin+t[:,:2].sum(1)-target).square().sum(-1))]:rows[prefix+'_'+metric]=v.reshape(128,4).sum(1).cpu().numpy()
                for n,q in qs.items():
                    pred=cross_response(x,z,mu,c,q,L,R,D,p)
                    for metric,diff in [('cross',pred-t[:,1]),('total',lin+t[:,0]+pred-target)]:rows[f'{prefix}_{n}_{metric}']=diff.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
        print('head',h,'complete',flush=True)
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={};contrasts={}
    for label in cfg['payloads']:
        for scale in cfg['scales']:
            prefix=f'{label}_s{scale}';total=lambda key:sum(rows[f'{prefix}_h{h}_{key}'] for h in cfg['heads']);results[prefix]={};contrasts[prefix]={}
            for metric in ['cross','total']:
                energy=total(metric+'_energy')
                for n in qs:
                    e=total(n+'_'+metric);results[prefix][n+'_'+metric]={'fidelity':float(1-e.sum()/energy.sum()),'document_bootstrap95':np.quantile(1-e[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
                for contrast,(n,b) in {'capacity64':('k64_optimized','k32_optimized'),'capacity32':('k32_optimized','k16_optimized'),'gain32':('k32_optimized','k32_weighted')}.items():
                    d=total(b+'_'+metric)-total(n+'_'+metric);contrasts[prefix][contrast+'_'+metric]={'gain':float(d.sum()/energy.sum()),'paired95':np.quantile(d[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
            results[prefix]['all_total']=float(1-total('all_total').sum()/total('total_energy').sum())
    gates={'capacity_gain':all(contrasts[f'actual_s{s}']['capacity64_cross']['gain']>=.1 and contrasts[f'actual_s{s}']['capacity64_cross']['paired95'][0]>0 for s in cfg['scales']),'compact80':all(results[f'actual_s{s}']['k64_optimized_cross']['fidelity']>=.8 for s in cfg['scales']),'replicated_gain32':all(contrasts[f'actual_s{s}']['gain32_cross']['gain']>=.05 and contrasts[f'actual_s{s}']['gain32_cross']['paired95'][0]>0 for s in cfg['scales']),'angle_transfer':all(results[f'angle_s{s}']['k64_optimized_cross']['fidelity']>=.5 for s in cfg['scales'])};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'models':models,'results':results,'contrasts':contrasts,'gates':gates,'head_replay':replay},indent=2));print('GATES',gates,flush=True)
if __name__=='__main__':main()
