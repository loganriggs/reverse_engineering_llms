#!/usr/bin/env python3
"""Full-output local source geometry with state/source mismatch controls."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources

ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_l8_context_response_geometry'

def source_jacobian(pre,left,right,down,source):
    dim=pre.shape[-1];rms=(pre.square().mean(-1,keepdim=True)+torch.finfo(pre.dtype).eps).sqrt();z=pre/rms
    l=z@left.T;r=z@right.T;write=(l*r)@down.T
    raw=torch.einsum('ol,nlk->nok',down,r[:,:,None]*(left@source)[None]+l[:,:,None]*(right@source)[None])
    return raw/rms[:,:,None]-2*write[:,:,None]*(z@source)[:,None,:]/(dim*rms[:,:,None])

def project(x,basis):
    if basis.ndim==2:return (x@basis.T)@basis
    return torch.einsum('nr,nrd->nd',torch.einsum('nd,nrd->nr',x,basis),basis)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());torch.manual_seed(cfg['seed'])
    model=load_full_model('cuda');model.embedding.float();model.blocks.float();corpus=np.load(ROOT/cfg['corpus'],mmap_mode='r')
    pre,hw,replay=collect_sources(model,corpus,cfg);block=model.blocks[8]
    left,right,down=[x.weight for x in [block.mlp.Left,block.mlp.Right,block.mlp.Down]]
    dim=pre.shape[-1];N=len(pre);native=block.mlp.Down(block.mlp.Left(F.rms_norm(pre,(dim,)))*block.mlp.Right(F.rms_norm(pre,(dim,))))
    bases={};tangents={};spectra={};orthogonality=0.;span_error=0.;derivative_error=0.;derivative_energy=0.
    for h in range(9):
        source=torch.linalg.qr(block.attn.c_proj.weight.reshape(dim,9,-1)[:,h]).Q
        span_error=max(span_error,float((hw[:,h]-(hw[:,h]@source)@source.T).norm()/hw[:,h].norm()))
        us=[];ss=[];pred=[]
        for offset in range(0,N,cfg['matrix_batch']):
            x=pre[offset:offset+cfg['matrix_batch']];J=source_jacobian(x,left,right,down,source)
            u,s,vh=torch.linalg.svd(J,full_matrices=False)
            # CUDA SVD may leave small orthogonality errors. Ordered QR preserves
            # every leading-column span while making coverage a true projection.
            u=torch.linalg.qr(u).Q
            us.append(u.transpose(1,2));ss.append(s)
            coefficients=hw[offset:offset+len(x),h]@source
            pred.append(-torch.einsum('ndr,nr->nd',J,coefficients))
            orthogonality=max(orthogonality,float((u.transpose(1,2)@u-torch.eye(128,device='cuda')).abs().max()))
        bases[h]=torch.cat(us);spectra[h]=torch.cat(ss);tangents[h]=torch.cat(pred)
        # Independent autograd JVP in double precision, one actual source direction per head.
        with torch.inference_mode(False),torch.enable_grad():
            px=pre[:1].detach().clone().double();direction=-hw[:1,h].detach().clone().double()
            L,R,D=[w.detach().clone().double() for w in [left,right,down]]
            # Preserve float32 RMS epsilon used by the native experiment.
            def fn(p):
                z=p/(p.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
                return ((z@L.T)*(z@R.T))@D.T
            _,jvp=torch.autograd.functional.jvp(fn,px,direction)
            derivative_error+=float((tangents[h][:1].double()-jvp).square().sum());derivative_energy+=float(jvp.square().sum())
        print('head geometry',h,'ready',flush=True)
    # Bases are selected only from state/weights; no finite-response target was used above.
    np.savez_compressed(STEM.with_name(STEM.name+'_singular_values.npz'),**{f'head{h}':s.cpu().numpy() for h,s in spectra.items()})
    global_file=np.load(STEM.parent/'bilin18_l8_response_population_frozen.npz')
    global_bases={r:torch.from_numpy(global_file[f'response_all{r}']).cuda() for r in cfg['ranks']}
    wrong_context=(torch.arange(N,device='cuda')+len(cfg['positions']))%N
    accum={};perhead={};rows={};spectral_capture={}
    for h in range(9):
        spectral_capture[str(h)]={str(r):float(spectra[h][:,:r].square().sum()/spectra[h].square().sum()) for r in cfg['ranks']}
        for scale in cfg['scales']:
            changed=F.rms_norm(pre-scale*hw[:,h],(dim,));target=block.mlp.Down(block.mlp.Left(changed)*block.mlp.Right(changed))-native
            tangent=scale*tangents[h];energy=target.square().sum(-1);key=f'{scale:g}'
            rows.setdefault(key+'_energy',np.zeros(cfg['rows']));rows[key+'_energy']+=energy.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
            entries={'full_tangent':tangent}
            coverages={}
            for rank in cfg['ranks']:
                charts={'matched':bases[h][:,:rank],'wrong_context':bases[h][wrong_context,:rank],
                        'wrong_source':bases[(h+1)%9][:,:rank],'global':global_bases[rank]}
                for family,basis in charts.items():
                    name=f'{family}{rank}';oracle=project(target,basis);prediction=project(tangent,basis)
                    coverages[name]=oracle.square().sum(-1)
                    entries[name]=prediction
            for name,prediction in entries.items():
                err=(prediction-target).square().sum(-1);rkey=key+'_'+name
                rows.setdefault(rkey+'_error',np.zeros(cfg['rows']));rows[rkey+'_error']+=err.reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                if name in coverages:
                    rows.setdefault(rkey+'_captured',np.zeros(cfg['rows']));rows[rkey+'_captured']+=coverages[name].reshape(cfg['rows'],-1).sum(1).cpu().numpy()
                perhead[f'h{h}_s{key}_{name}']={'prediction_fidelity':float(1-err.sum()/energy.sum()),'coverage':float(coverages[name].sum()/energy.sum()) if name in coverages else None}
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows)
    rng=np.random.default_rng(cfg['seed']);ix=rng.integers(cfg['rows'],size=(2000,cfg['rows']));results={};contrasts={}
    for scale in cfg['scales']:
        key=f'{scale:g}';energy=rows[key+'_energy'];results[key]={};contrasts[key]={}
        for name in ['full_tangent']+[f'{family}{r}' for r in cfg['ranks'] for family in cfg['families']]:
            rkey=key+'_'+name;err=rows[rkey+'_error'];result={'prediction_fidelity':float(1-err.sum()/energy.sum()),'prediction_95':np.quantile(1-err[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
            if rkey+'_captured' in rows:result['coverage']=float(rows[rkey+'_captured'].sum()/energy.sum())
            results[key][name]=result
        for control in ['wrong_context16','wrong_source16','global16']:
            difference=rows[key+'_matched16_captured']-rows[key+'_'+control+'_captured']
            contrasts[key][control]={'coverage_gain':float(difference.sum()/energy.sum()),'paired_row_bootstrap_95':np.quantile(difference[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
    derivative_relative=(derivative_error/derivative_energy)**.5
    gates={'locally_compact':all(r['matched16']['coverage']>=.8 and r['matched16']['coverage']-r['global16']['coverage']>=.2 for r in results.values()),
           'context_specific':all(r['matched16']['coverage']-r['wrong_context16']['coverage']>=.1 for r in results.values()),
           'source_specific':all(r['matched16']['coverage']-r['wrong_source16']['coverage']>=.1 for r in results.values()),
           'small_change_prediction':results['0.05']['full_tangent']['prediction_fidelity']>=.99 and derivative_relative<=1e-4}
    output={'config':cfg,'gates':gates,'head_replay':replay,'source_span_relative_error':span_error,'basis_orthogonality':orthogonality,'analytic_vs_autograd_jvp_relative_error':derivative_relative,'results':results,'head_results':perhead,'isotropic_tangent_spectral_capture':spectral_capture,'contrasts':contrasts}
    STEM.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=['# Context-conditioned full-output response geometry','','Bases depend on unperturbed state and native weights, not on the evaluated finite response. Oracle coverage and tangent prediction are distinct.','','| Basis | Small-edit coverage | Small-edit prediction | Full-edit coverage | Full-edit prediction |','|---|---:|---:|---:|---:|']
    for name in results['0.05']:
        small=results['0.05'][name];full=results['1'][name]
        sc=f"{small['coverage']:.3f}" if 'coverage' in small else '—';fc=f"{full['coverage']:.3f}" if 'coverage' in full else '—'
        lines.append(f"| {name} | {sc} | {small['prediction_fidelity']:.3f} | {fc} | {full['prediction_fidelity']:.3f} |")
    lines+=['',f'Locked gates: {gates}. Analytic derivative relative error versus independent autograd JVP: {derivative_relative:.3g}.','',cfg['cost'],'','The native full tangent supplies coefficients for every projected predictor; wrong-chart controls vary only the output basis. No projected predictor uses the changed output. Rank128 corresponds to the full source-local tangent span, not a128-dimensional global code.']
    STEM.with_name(STEM.name.upper()+'.md').write_text('\n'.join(lines)+'\n')
    print(gates,flush=True)

if __name__=='__main__':main()
