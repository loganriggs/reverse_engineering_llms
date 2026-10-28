"""H266 fixed-width input basis comparison for the remaining residual polynomial."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_response_capacity import response_solutions
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_response_input_bases'

def residual_pullback(x,v,L,R,D,p):
    vd=v@D;native=(vd*(x@R.T)[:,None,:])@L+(vd*(x@L.T)[:,None,:])@R
    z=x@p['reads']-p['offset'];t=torch.einsum('nk,okr->nor',z,p['eigenvectors']);deriv=2*torch.einsum('nor,or,okr->nok',t,p['eigenvalues'],p['eigenvectors'])+p['linear'][None]
    compiled=torch.einsum('npo,nok->npk',v@p['outputs'].T,deriv)@p['reads'].T
    return native-compiled

def features(x,linear,qbasis,mean):
    delta=x-mean;z=delta@qbasis;ij=torch.triu_indices(z.shape[1],z.shape[1],device=x.device);products=z[:,ij[0]]*z[:,ij[1]]*torch.where(ij[0]==ij[1],1.,2**.5)
    return torch.cat([delta@linear,products],1)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');old=load_program(ART/'bilin18_l17_residual_feature_objective_frozen.npz','cuda');linear=old['basis'][:,:624];mean=old['input_mean'];pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));cut=cfg['train_documents']*4
    def error(z):return B(z,z,L,R,D)-forward_normalized(z,p)
    y=error(x);zs={h:F.rms_norm(pre-heads[:,h],(1152,)) for h in cfg['heads']};dy={h:error(z)-y for h,z in zs.items()};source_cov=sum((z[:cut]-x[:cut]).double().T@(z[:cut]-x[:cut]).double() for z in zs.values());_,q=torch.linalg.eigh(source_cov);bases={'natural':old['basis'][:,:32],'source':q[:,-32:].flip(1).float()};jac_cov=torch.zeros(1152,1152,device='cuda',dtype=torch.float64);gen=torch.Generator(device='cuda').manual_seed(cfg['seed'])
    for i in range(0,cut,16):
        xx=x[i:i+16];v=torch.randn(len(xx),cfg['gradient_probes'],1152,generator=gen,device='cuda');g=residual_pullback(xx,v,L,R,D,p).reshape(-1,1152).double();jac_cov+=g.T@g
    ev,q=torch.linalg.eigh(jac_cov);bases['residual_jacobian']=q[:,-32:].flip(1).float();diagnostic={name:{'orthogonality_error':float((q.T@q-torch.eye(32,device='cuda')).abs().max()),'fraction_basis_energy_outside_linear624':float(1-(linear.T@q).square().sum()/32)} for name,q in bases.items()};diagnostic['jacobian_top32_trace_fraction']=float(ev[-32:].sum()/ev.sum());print('BASES',diagnostic,flush=True)
    ty=torch.cat([v[:cut] for v in dy.values()]);vy=torch.cat([v[cut:] for v in dy.values()]);normalizers={};fitted={};selected={};audits={};validation={};frozen={'linear_basis':linear.cpu().numpy(),'input_mean':mean.cpu().numpy()}
    for name,q in bases.items():
        raw=features(x,linear,q,mean)
        if name=='natural':fm=old['quadratic1152_mean'];fs=old['quadratic1152_std']
        else:fm=raw[:cut].mean(0);fs=raw[:cut].std(0,unbiased=False).clamp_min(1e-6)
        f=(raw-fm)/fs;normalizers[name]=(fm,fs);df={h:(features(z,linear,q,mean)-fm)/fs-f for h,z in zs.items()};tf=torch.cat([v[:cut] for v in df.values()]);vf=torch.cat([v[cut:] for v in df.values()]);solutions,audit=response_solutions(tf,ty,cfg['ridge_strengths']);audits[name]=audit
        frozen[name+'_basis']=q.cpu().numpy();frozen[name+'_mean']=fm.cpu().numpy();frozen[name+'_std']=fs.cpu().numpy()
        for alpha,w64 in solutions.items():
            key=f'{name}_a{alpha}';w=w64.float();b=y[:cut].mean(0)-f[:cut].mean(0)@w;frozen[key]=torch.cat([w,b[None]],0).cpu().numpy();fitted[key]=w;validation[key]=float(1-(vf@w-vy).square().sum()/vy.square().sum());print(key,'train',audit['fits'][str(alpha)]['training_fidelity'],'val',validation[key],flush=True)
        selected[name]=max([f'{name}_a{alpha}' for alpha in cfg['ridge_strengths']],key=lambda k:validation[k])
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'selected':selected,'audits':audits,'validation':validation,'basis_diagnostics':diagnostic},indent=2));print('FROZEN',selected,flush=True)
    pre,heads,replay=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));y=error(x);intact={name:(features(x,linear,q,mean)-normalizers[name][0])/normalizers[name][1] for name,q in bases.items()};rows={};gen.manual_seed(cfg['seed']+1)
    for h in cfg['heads']:
        source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;actual=heads[:,h];angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
        for payload,pay in [('actual',actual),('angle',angle)]:
            for scale in cfg['scales']:
                z=F.rms_norm(pre-scale*pay,(1152,));target=error(z)-y;prefix=f'{payload}_s{scale}_h{h}';rows[prefix+'_energy']=target.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
                for name,q in bases.items():
                    fm,fs=normalizers[name];df=(features(z,linear,q,mean)-fm)/fs-intact[name]
                    for role,key in [('selected',selected[name]),('unregularized',name+'_a0')]:rows[prefix+'_'+name+'_'+role]=(df@fitted[key]-target).square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
    results={};contrasts={};ix=np.random.default_rng(cfg['seed']+2).integers(128,size=(2000,128))
    for payload in cfg['payloads']:
        for scale in cfg['scales']:
            prefix=f'{payload}_s{scale}';total=lambda m:sum(rows[f'{prefix}_h{h}_{m}'] for h in cfg['heads']);energy=total('energy');results[prefix]={};contrasts[prefix]={}
            for name in bases:
                for role in ['selected','unregularized']:
                    e=total(name+'_'+role);results[prefix][name+'_'+role]={'fidelity':float(1-e.sum()/energy.sum()),'document_bootstrap95':np.quantile(1-e[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
                if name!='natural':
                    delta=total('natural_selected')-total(name+'_selected');contrasts[prefix][name]={'gain':float(delta.sum()/energy.sum()),'paired_document_bootstrap95':np.quantile(delta[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
    gates={label:all(contrasts[f'actual_s{s}'][name]['gain']>=.05 and contrasts[f'actual_s{s}'][name]['paired_document_bootstrap95'][0]>0 for s in cfg['scales']) for name,label in [('source','source_basis_gain'),('residual_jacobian','jacobian_basis_gain')]};gates['half_response']=all(results[f'actual_s{s}']['residual_jacobian_selected']['fidelity']>=.5 for s in cfg['scales'])
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'selected':selected,'audits':audits,'validation':validation,'basis_diagnostics':diagnostic,'results':results,'contrasts':contrasts,'gates':gates,'head_replay':replay},indent=2));print('RESULTS',results,'CONTRASTS',contrasts,'GATES',gates,flush=True)
if __name__=='__main__':main()
