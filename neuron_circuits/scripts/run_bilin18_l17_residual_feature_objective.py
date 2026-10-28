"""H259 fixed-feature/conditional-metric factorial, independent document evaluation."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_residual_restoration import score_gradients
STEM=ART/'bilin18_l17_residual_feature_objective'

def metric_apply(v,g,scale):
    if g is None:return v
    return .5*(v+scale*torch.einsum('nk,nkd->nd',torch.einsum('nd,nkd->nk',v,g),g)/g.shape[1])

def fit_metric(f,y,alpha,g=None,maxiter=300,tol=1e-5):
    n,d=y.shape;x=torch.cat([f,torch.ones(len(f),1,device=f.device,dtype=f.dtype)],1);lam=alpha*f.square().mean();pen=torch.ones(x.shape[1],1,device=x.device,dtype=x.dtype);pen[-1]=0
    scale=d/g.square().sum(-1).mean() if g is not None else 0
    def op(w):return x.T@metric_apply(x@w,g,scale)/n+lam*pen*w
    rhs=x.T@metric_apply(y,g,scale)/n
    # Exact diagonal of the normal operator (Jacobi preconditioning).
    md=torch.ones_like(y) if g is None else .5*(1+scale*g.square().mean(1))
    diag=x.square().T@md/n+lam*pen;diag=diag.clamp_min(1e-12)
    w=torch.zeros_like(rhs);res=rhs.clone();z=res/diag;p=z.clone();rz=(res*z).sum();initial=rhs.norm();history=[]
    for iteration in range(maxiter):
        ap=op(p);step=rz/(p*ap).sum();w+=step*p;res-=step*ap;rel=float(res.norm()/initial)
        if iteration%25==0:history.append([iteration+1,rel])
        if rel<tol:break
        z=res/diag;new=(res*z).sum();p=z+(new/rz)*p;rz=new
    exact=float((op(w)-rhs).norm()/initial)
    return w,{'iterations':iteration+1,'recursive_relative_residual':rel,'exact_relative_residual':exact,'converged':exact<5*tol,'history':history}

def raw_features(x,basis,mean,family):
    z=(x-mean)@basis
    if family=='affine1152':return z
    if family=='affine624':return z[:,:624]
    ij=torch.triu_indices(32,32,device=x.device);q=z[:,ij[0]]*z[:,ij[1]];q=q*torch.where(ij[0]==ij[1],1.,2**.5)
    return torch.cat([z[:,:624],q],1)

@torch.inference_mode()
def main():
    torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];program=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda')
    def panel(path,n):
        corpus=np.load(ROOT/path,mmap_mode='r');pre,_,_=collect_sources(model,corpus,{**cfg,'layer':17,'rows':n});x=F.rms_norm(pre,(1152,));pred=forward_normalized(x,program);state=pre+pred+block.mlp.Down_bias;error=B(x,x,L,R,D)-pred;labels=torch.from_numpy(np.array(corpus[:n,np.array(cfg['positions'])+1],dtype=np.int64).flatten()).cuda();return x,state,error,labels
    def logits(s):return 30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)
    def score(state,error,labels,c):
        rows={k:[] for k in ['kl','ce_delta','top1']}
        for i in range(0,len(state),16):
            sl=slice(i,i+16);p=logits(state[sl]+error[sl]);q=logits(state[sl]+c[sl]);rows['kl'].append((p.softmax(-1)*(p.log_softmax(-1)-q.log_softmax(-1))).sum(-1));rows['ce_delta'].append(F.cross_entropy(q,labels[sl],reduction='none')-F.cross_entropy(p,labels[sl],reduction='none'));rows['top1'].append((p.argmax(-1)==q.argmax(-1)).float())
        return {k:torch.cat(v).reshape(-1,4).sum(1).cpu().numpy() for k,v in rows.items()}
    x,state,error,labels=panel(cfg['discovery'],1024);cut=cfg['train_documents']*4;mean=x[:cut].mean(0);_,basis=torch.linalg.eigh((x[:cut]-mean).double().T@(x[:cut]-mean).double());basis=basis.flip(1).float();grad=[];gen=torch.Generator(device='cuda').manual_seed(cfg['seed'])
    for i in range(0,cut,16):
        st=state[i:i+16]+error[i:i+16];p=logits(st).softmax(-1);ids=torch.multinomial(p,cfg['score_samples'],replacement=True,generator=gen);g,_=score_gradients(st,model.lm_head.weight,ids);grad.append(g)
    grad=torch.cat(grad);fitted={};stats={};validation={};diagnostics={};frozen={'basis':basis.cpu().numpy(),'input_mean':mean.cpu().numpy()}
    for family in cfg['families']:
        raw=raw_features(x,basis,mean,family);fm=raw[:cut].mean(0);fs=raw[:cut].std(0,unbiased=False).clamp_min(1e-6);features=(raw-fm)/fs;stats[family]=(fm,fs);frozen[family+'_mean']=fm.cpu().numpy();frozen[family+'_std']=fs.cpu().numpy()
        for objective in cfg['objectives']:
            for alpha in cfg['ridge_strengths']:
                key=f'{family}_{objective}_{alpha}';w,diag=fit_metric(features[:cut],error[:cut],alpha,grad if objective=='sensitivity' else None,maxiter=cfg.get('solver_maxiter',300));fitted[key]=w;diagnostics[key]=diag;frozen[key]=w.cpu().numpy();correction=torch.cat([features[cut:],torch.ones(len(features)-cut,1,device='cuda')],1)@w;validation[key]=float(score(state[cut:],error[cut:],labels[cut:],correction)['kl'].mean()/4);print(key,diag['iterations'],diag['exact_relative_residual'],validation[key],flush=True)
    selected={f'{f}_{obj}':min([f'{f}_{obj}_{alpha}' for alpha in cfg['ridge_strengths']],key=lambda k:validation[k]) for f in cfg['families'] for obj in cfg['objectives']}
    np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),**frozen);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'selected':selected,'validation_kl':validation,'solver':diagnostics},indent=2));print('FROZEN',selected,flush=True)
    x,state,error,labels=panel(cfg['corpus'],128);rows={};results={};corrections={'compiled':torch.zeros_like(error),'oracle':error}
    for family in cfg['families']:
        fm,fs=stats[family];f=(raw_features(x,basis,mean,family)-fm)/fs;f=torch.cat([f,torch.ones(len(f),1,device='cuda')],1)
        for name,w in fitted.items():
            if name.startswith(family+'_'):corrections[name]=f@w
    for name,c in corrections.items():
        metrics=score(state,error,labels,c)
        for k,v in metrics.items():rows[name+'_'+k]=v
        results[name]={'kl':float(metrics['kl'].mean()/4),'ce_delta':float(metrics['ce_delta'].mean()/4),'top1':float(metrics['top1'].mean()/4),'energy_recovery':float(1-(error-c).square().sum()/error.square().sum())}
    for v in results.values():v['kl_recovery']=1-v['kl']/results['compiled']['kl']
    ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));contrasts={}
    def compare(a,b):
        d=rows[b+'_kl']-rows[a+'_kl'];base=rows['compiled_kl'];return {'recovery_gain':float(d.sum()/base.sum()),'paired_document_bootstrap95':np.quantile(d[ix].sum(1)/base[ix].sum(1),[.025,.975]).tolist()}
    for obj in cfg['objectives']:
        contrasts['quadratic_'+obj]=compare(selected['quadratic1152_'+obj],selected['affine1152_'+obj]);contrasts['interactions_'+obj]=compare(selected['quadratic1152_'+obj],selected['affine624_'+obj])
    for fam in ['affine1152','quadratic1152']:contrasts['objective_'+fam]=compare(selected[fam+'_sensitivity'],selected[fam+'_energy'])
    gates={kind+'_benefit':all(v['recovery_gain']>=.05 and v['paired_document_bootstrap95'][0]>0 for k,v in contrasts.items() if k.startswith(kind+'_')) for kind in ['quadratic','objective']};gates['behavioral_repair']=any(results[n]['kl_recovery']>=.8 and results[n]['ce_delta']<=.02 and results[n]['top1']>=.98 for k,n in selected.items() if not k.startswith('affine624'))
    np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'selected':selected,'validation_kl':validation,'solver':diagnostics,'results':results,'contrasts':contrasts,'gates':gates},indent=2));print('RESULTS',json.dumps({k:results[v] for k,v in selected.items()}),'CONTRASTS',contrasts,'GATES',gates,flush=True)
if __name__=='__main__':main()
