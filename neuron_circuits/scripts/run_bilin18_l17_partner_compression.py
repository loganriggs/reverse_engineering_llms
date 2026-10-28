"""H271 exact cross-core compression under tensor/data-weighted/random partner bases."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_residual_tensor_partners import residual_bilinear_matrix
from run_bilin18_l17_residual_term_accounting import residual_bilinear,mean_linear
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
STEM=ART/'bilin18_l17_partner_compression'
def cross_response(x,z,mean,core,partner,L,R,D,p):
    a=x-mean;b=z-mean;ca=(a@core)@core.T;cb=(b@core)@core.T;pa=(a@partner)@partner.T;pb=(b@partner)@partner.T
    return residual_bilinear(ca+cb,pb-pa,L,R,D,p)+residual_bilinear(cb-ca,pa+pb,L,R,D,p)
def core_metric(x,z,mean,core):
    a=x-mean;b=z-mean;ca=a@core;cb=b@core;ra=a-ca@core.T;rb=b-cb@core.T
    u=ca+cb;v=cb-ca;w1=(rb-ra).square().sum(1);w2=(ra+rb).square().sum(1)
    return u.T@(u*w1[:,None])+v.T@(v*w2[:,None])
@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();L,R,D=[m.weight.double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda',torch.float64);f=load_program(ART/'bilin18_l17_core_width_accounting_bases.npz','cuda',torch.float64);mean=f['input_mean'];cores={n:torch.linalg.qr(f[n][:,:32]).Q for n in cfg['families']};pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':768});x=F.rms_norm(pre,(1152,)).double();zs=[F.rms_norm(pre-heads[:,h],(1152,)).double() for h in cfg['heads']];partners={};diag={};saved={'mean':mean.cpu().numpy()};gen=torch.Generator(device='cuda').manual_seed(cfg['seed'])
    for name,c in cores.items():
        metric=sum(core_metric(x,z,mean,c) for z in zs);metric/=torch.trace(metric)/32;ev,v=torch.linalg.eigh(metric);weighted=c@(v*ev.clamp_min(0).sqrt()[None]);proj=torch.eye(1152,device='cuda',dtype=torch.float64)-c@c.T;saved[name+'_core']=c.cpu().numpy();diag[name]={};partners[name]={}
        for method,cs in [('tensor',c),('weighted',weighted)]:
            gram=torch.zeros(1152,1152,device='cuda',dtype=torch.float64)
            for cc in cs.T:
                m=residual_bilinear_matrix(cc,L,R,D,p);gram+=m.T@m
            gram=proj@gram@proj;ev,q=torch.linalg.eigh((gram+gram.T)/2);partners[name][method]=q[:,-64:].flip(1);diag[name][method]={'top64_metric_fraction':float(ev[-64:].sum()/ev.sum())}
        for n in range(3):partners[name][f'random{n}']=torch.linalg.qr(proj@torch.randn(1152,64,device='cuda',dtype=torch.float64,generator=gen)).Q
        for method,q in partners[name].items():
            saved[name+'_'+method]=q.cpu().numpy();diag[name].setdefault(method,{}).update({'overlap':float((c.T@q).abs().max()),'orthogonality':float((q.T@q-torch.eye(64,device='cuda')).abs().max())})
        print(name,'partners frozen',flush=True)
    np.savez_compressed(STEM.with_name(STEM.name+'_bases.npz'),**saved);STEM.with_name(STEM.name+'_basis_diagnostics.json').write_text(json.dumps(diag,indent=2));pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double();err=lambda z:B(z,z,L,R,D)-forward_normalized(z,p);e0=err(x);rows={};max_replay=0
    for h in cfg['heads']:
        for scale in cfg['scales']:
            z=F.rms_norm(pre-scale*heads[:,h],(1152,)).double();target=err(z)-e0;lin=mean_linear(z-x,mean,L,R,D,p);prefix=f's{scale}_h{h}';rows[prefix+'_energy']=target.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
            for name,c in cores.items():
                t=quadratic_terms(x,z,mean,c,L,R,D,p);max_replay=max(max_replay,float((lin+t.sum(1)-target).norm()/target.norm()));base=lin+t[:,0];cross=t[:,1]
                for key,val in [('cross_energy',cross.square().sum(-1)),('within_error',(base-target).square().sum(-1)),('all_error',(base+cross-target).square().sum(-1))]:rows[f'{prefix}_{name}_{key}']=val.reshape(128,4).sum(1).cpu().numpy()
                for method,q in partners[name].items():
                    for k in cfg['widths']:
                        pred=cross_response(x,z,mean,c,q[:,:k],L,R,D,p);key=f'{prefix}_{name}_{method}{k}'
                        for metric,val in [('cross_error',(pred-cross).square().sum(-1)),('total_error',(base+pred-target).square().sum(-1))]:rows[key+'_'+metric]=val.reshape(128,4).sum(1).cpu().numpy()
        print('head',h,'complete',flush=True)
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={};contrasts={}
    for scale in cfg['scales']:
        prefix=f's{scale}';total=lambda suffix:sum(rows[f'{prefix}_h{h}_{suffix}'] for h in cfg['heads']);energy=total('energy');results[prefix]={};contrasts[prefix]={}
        for name in cores:
            ce=total(name+'_cross_energy');res={};results[prefix][name]=res
            for method in partners[name]:
                for k in cfg['widths']:
                    key=f'{method}{k}';er=total(f'{name}_{key}_cross_error');te=total(f'{name}_{key}_total_error');res[key]={'cross_fidelity':float(1-er.sum()/ce.sum()),'cross_bootstrap95':np.quantile(1-er[ix].sum(1)/ce[ix].sum(1),[.025,.975]).tolist(),'total_fidelity':float(1-te.sum()/energy.sum())}
            res['within_total']=float(1-total(name+'_within_error').sum()/energy.sum());res['all_total']=float(1-total(name+'_all_error').sum()/energy.sum())
            for other in ['tensor','random0','random1','random2']:
                d=total(f'{name}_{other}32_cross_error')-total(f'{name}_weighted32_cross_error');contrasts[prefix][name+'_'+other]={'gain':float(d.sum()/ce.sum()),'paired95':np.quantile(d[ix].sum(1)/ce[ix].sum(1),[.025,.975]).tolist()}
    gates={'compact32':any(all(results[f's{s}'][n][m+'32']['cross_fidelity']>=.8 for s in cfg['scales']) for n in cores for m in ['tensor','weighted']),'weighted_gain':all(contrasts[f's{s}'][n+'_tensor']['gain']>=.05 and contrasts[f's{s}'][n+'_tensor']['paired95'][0]>0 for s in cfg['scales'] for n in cores),'weighted_specificity':all(contrasts[f's{s}'][n+'_'+m]['gain']>=.05 and contrasts[f's{s}'][n+'_'+m]['paired95'][0]>0 for s in cfg['scales'] for n in cores for m in ['random0','random1','random2'])};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'contrasts':contrasts,'gates':gates,'basis_diagnostics':diag,'max_replay':max_replay,'head_replay':replay},indent=2));print('GATES',gates,flush=True)
if __name__=='__main__':main()
