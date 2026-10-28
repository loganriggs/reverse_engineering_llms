"""H267 residual-tensor mixed partners versus random complement and within-space products."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_response_capacity import response_solutions
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_residual_tensor_partners'

def residual_bilinear_matrix(c,L,R,D,p):
    native=.5*D@((R@c)[:,None]*L+(L@c)[:,None]*R)
    modes=torch.einsum('k,okr->or',c@p['reads'],p['eigenvectors'])
    rows=torch.einsum('or,or,okr->ok',modes,p['eigenvalues'],p['eigenvectors'])
    return native-p['outputs'].T@rows@p['reads'].T

def products(z):
    ij=torch.triu_indices(z.shape[-1],z.shape[-1],device=z.device)
    return z[:,ij[0]]*z[:,ij[1]]*torch.where(ij[0]==ij[1],1.,2**.5)

def features(x,linear,mean,core,within,partner,name):
    delta=x-mean
    if name=='within32':q=products(delta@within)
    else:
        z=delta@core;q=products(z)
        if partner is not None:q=torch.cat([q,(2**.5*z[:,:,None]*(delta@partner)[:,None,:]).flatten(1)],1)
    return torch.cat([delta@linear,q],1)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');old=load_program(ART/'bilin18_l17_response_input_bases_frozen.npz','cuda');linear=old['linear_basis'];mean=old['input_mean'];within=old['residual_jacobian_basis'];core=within[:,:16]
    cc=core.double();pd={k:v.double() for k,v in p.items()};gram=torch.zeros(1152,1152,device='cuda',dtype=torch.float64)
    for c in cc.T:
        m=residual_bilinear_matrix(c,L.double(),R.double(),D.double(),pd);gram+=m.T@m
    projection=torch.eye(1152,device='cuda',dtype=torch.float64)-cc@cc.T;restricted=projection@gram@projection;ev,q=torch.linalg.eigh((restricted+restricted.T)/2);tensor=q[:,-24:].flip(1).float();partners={'within32':None,'core16':None,'tensor':tensor};gen=torch.Generator(device='cuda').manual_seed(cfg['seed'])
    for k in range(3):
        z=torch.randn(1152,24,generator=gen,device='cuda');z=z-core@(core.T@z);partners[f'random{k}']=torch.linalg.qr(z).Q
    diagnostics={}
    for name,q in partners.items():
        if q is not None:diagnostics[name]={'core_overlap_max':float((core.T@q).abs().max()),'orthogonality_error':float((q.T@q-torch.eye(24,device='cuda')).abs().max()),'restricted_tensor_energy_fraction':float(torch.trace(q.double().T@gram@q.double())/torch.trace(restricted))}
    print('PARTNERS',diagnostics,flush=True)
    pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));cut=cfg['train_documents']*4
    def error(z):return B(z,z,L,R,D)-forward_normalized(z,p)
    y=error(x);zs={h:F.rms_norm(pre-heads[:,h],(1152,)) for h in cfg['heads']};dy={h:error(z)-y for h,z in zs.items()};ty=torch.cat([v[:cut] for v in dy.values()]);vy=torch.cat([v[cut:] for v in dy.values()]);norms={};fitted={};selected={};audits={};validation={};frozen={'linear_basis':linear.cpu().numpy(),'input_mean':mean.cpu().numpy(),'core':core.cpu().numpy(),'within':within.cpu().numpy()}
    for name,q in partners.items():
        raw=features(x,linear,mean,core,within,q,name)
        if name=='within32':fm=old['residual_jacobian_mean'];fs=old['residual_jacobian_std']
        else:fm=raw[:cut].mean(0);fs=raw[:cut].std(0,unbiased=False).clamp_min(1e-6)
        f=(raw-fm)/fs;norms[name]=(fm,fs);frozen[name+'_mean']=fm.cpu().numpy();frozen[name+'_std']=fs.cpu().numpy()
        if q is not None:frozen[name+'_partner']=q.cpu().numpy()
        df={h:(features(z,linear,mean,core,within,q,name)-fm)/fs-f for h,z in zs.items()};tf=torch.cat([v[:cut] for v in df.values()]);vf=torch.cat([v[cut:] for v in df.values()]);solutions,audit=response_solutions(tf,ty,cfg['ridge_strengths']);audits[name]=audit
        for alpha,w64 in solutions.items():
            key=f'{name}_a{alpha}';w=w64.float();b=y[:cut].mean(0)-f[:cut].mean(0)@w;frozen[key]=torch.cat([w,b[None]],0).cpu().numpy();fitted[key]=w;validation[key]=float(1-(vf@w-vy).square().sum()/vy.square().sum());print(key,'train',audit['fits'][str(alpha)]['training_fidelity'],'val',validation[key],flush=True)
        selected[name]=max([f'{name}_a{alpha}' for alpha in cfg['ridge_strengths']],key=lambda k:validation[k])
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'selected':selected,'audits':audits,'validation':validation,'partner_diagnostics':diagnostics},indent=2));print('FROZEN',selected,flush=True)
    pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));y=error(x);intact={name:(features(x,linear,mean,core,within,q,name)-norms[name][0])/norms[name][1] for name,q in partners.items()};rows={};gen.manual_seed(cfg['seed']+1)
    for h in cfg['heads']:
        source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;actual=heads[:,h];angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
        for payload,pay in [('actual',actual),('angle',angle)]:
            for scale in cfg['scales']:
                z=F.rms_norm(pre-scale*pay,(1152,));target=error(z)-y;prefix=f'{payload}_s{scale}_h{h}';rows[prefix+'_energy']=target.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
                for name,q in partners.items():
                    fm,fs=norms[name];df=(features(z,linear,mean,core,within,q,name)-fm)/fs-intact[name]
                    for role,key in [('selected',selected[name]),('unregularized',name+'_a0')]:rows[prefix+'_'+name+'_'+role]=(df@fitted[key]-target).square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
    results={};contrasts={};ix=np.random.default_rng(cfg['seed']+2).integers(128,size=(2000,128))
    for payload in cfg['payloads']:
        for scale in cfg['scales']:
            prefix=f'{payload}_s{scale}';total=lambda m:sum(rows[f'{prefix}_h{h}_{m}'] for h in cfg['heads']);energy=total('energy');results[prefix]={};contrasts[prefix]={}
            for name in partners:
                for role in ['selected','unregularized']:
                    e=total(name+'_'+role);results[prefix][name+'_'+role]={'fidelity':float(1-e.sum()/energy.sum()),'document_bootstrap95':np.quantile(1-e[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
                if name!='tensor':
                    delta=total(name+'_selected')-total('tensor_selected');contrasts[prefix][name]={'tensor_gain':float(delta.sum()/energy.sum()),'paired_document_bootstrap95':np.quantile(delta[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
    def passes(names):return all(contrasts[f'actual_s{s}'][n]['tensor_gain']>=.05 and contrasts[f'actual_s{s}'][n]['paired_document_bootstrap95'][0]>0 for s in cfg['scales'] for n in names)
    gates={'partner_specificity':passes(['random0','random1','random2']),'mixed_advantage':passes(['within32']),'half_response':all(results[f'actual_s{s}']['tensor_selected']['fidelity']>=.5 for s in cfg['scales'])};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'selected':selected,'audits':audits,'validation':validation,'partner_diagnostics':diagnostics,'results':results,'contrasts':contrasts,'gates':gates,'head_replay':replay},indent=2));print('GATES',gates,'RESULTS',results,'CONTRASTS',contrasts,flush=True)
if __name__=='__main__':main()
