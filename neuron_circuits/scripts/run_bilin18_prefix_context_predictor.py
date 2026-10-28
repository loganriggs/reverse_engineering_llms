"""H605 native-context prediction of prefix effects, without evaluation dc."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_context_predictor'

def fit_predictors(z,y):
 z=z.double();y=y.double();zm=z.mean(0);ym=y.mean(0);Z=z-zm;Y=y-ym;beta=(Z*Y).sum()/Z.square().sum();G=Z@Z.T;lam=.1*G.trace()/len(z);system=G+lam*torch.eye(len(z),dtype=torch.float64);coef=torch.linalg.solve(system,Y);icoef=torch.linalg.solve(system,Y-beta*Z)
 with torch.inference_mode(False),torch.enable_grad():
  zz=z.clone();target=(z+y).clone();u=ym.clone().requires_grad_(True);opt=torch.optim.Adam([u],lr=.03)
  for step in range(500):
   opt.zero_grad();v=zz+u;pred=v/v.norm(dim=-1,keepdim=True)*(z.shape[1]**.5);loss=(pred-target).square().mean();loss.backward();opt.step()
  v=zz+u;loss=(v/v.norm(dim=-1,keepdim=True)*(z.shape[1]**.5)-target).square().mean();grad=torch.autograd.grad(loss,u)[0];u=u.detach().clone();gradnorm=float(grad.norm())
 params=dict(zmean=zm,ymean=ym,Z=Z,beta=beta,coef=coef,icoef=icoef,shift=u,lambda_value=float(lam),shift_gradient_norm=gradnorm,normal_error=float((system@coef-Y).norm()/Y.norm()))
 return params

def predict(p,z,mode):
 z=z.double();dz=z-p['zmean']
 if mode=='mean':return p['ymean'].float()
 if mode=='isotropic':return (p['ymean']+p['beta']*dz).float()
 if mode=='ridge':return (p['ymean']+(dz@p['Z'].T)@p['coef']).float()
 if mode=='isotropic_ridge':return (p['ymean']+p['beta']*dz+(dz@p['Z'].T)@p['icoef']).float()
 if mode=='normalized_shift':
  v=z+p['shift'];return (v/v.norm()*(len(z)**.5)-z).float()
 raise ValueError(mode)

@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_metric_wording_edit_factorial_captures.pt';d=torch.load(src,weights_only=False);ex=d['examples'];families=sorted({e['family'] for e in ex.values()});cfg=dict(experiment='H605',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='Same five leave-task/new-wording folds as H604. Fit dc from native normalized MLP11 input z on 32 other-task old-wording endpoints. Fixed mean, isotropic affine change, ridge, isotropic plus ridge residual, and common shift followed by RMS normalization. Same-task ridge/normalized-shift diagnostics fit eight old-wording endpoints, not task-held-out. No evaluation dc used for any predictive coefficients. All use actual e to execute T. Reused panel, adaptive not independent confirmation.',ridge_lambda_rule='.1 times mean diagonal of centered training Gram matrix; fixed before evaluation, no sweep',shift_fit='500 Adam steps, lr .03, float64, training-only normalized-input MSE, initialized at mean dc. No evaluation early stopping.',scope='No task-label input for shared models. Native context can encode format/task; this does not disentangle them. Low-dimensional parameterizations do not identify semantic units. Each fold is a separate cross-validation model.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));folds={}
 for fam in families:
  train=[k for k,e in ex.items() if k[1]<2 and e['template']==1 and e['family']!=fam];own=[k for k,e in ex.items() if k[1]<2 and e['template']==1 and e['family']==fam];test=[k for k,e in ex.items() if k[1]<2 and e['template']==3 and e['family']==fam];assert len(train)==32 and len(own)==len(test)==8
  def contexts(kk):return torch.stack([d['inputs'][k[0],k[1]+2]-d['inputs'][k] for k in kk])
  X=contexts(train);Y=contexts(own);zt=torch.stack([d['inputs'][k] for k in train]);zo=torch.stack([d['inputs'][k] for k in own]);shared=fit_predictors(zt,X);specific=fit_predictors(zo,Y);training_errors={}
  for mode in ['mean','isotropic','ridge','isotropic_ridge','normalized_shift']:
   pp=torch.stack([predict(shared,z,mode) for z in zt]);training_errors[mode]=float((pp-X).norm()/X.norm())
  folds[fam]=dict(train=train,own=own,test=test,shared=shared,specific=specific,training_errors=training_errors)

 fitted=ART/f'{STEM}_locked_folds.pt';torch.save(dict(folds=folds,config=cfg),fitted);fh=hashlib.sha256(fitted.read_bytes()).hexdigest();print('FROZEN',len(folds),'folds',flush=True)
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 m=model.blocks[11].mlp
 def T(c,e):
  c=c.cuda();e=e.cuda();v=m.Left(c)*m.Right(e)+m.Left(e)*m.Right(c);return (m.Down(v)-m.Down(torch.zeros_like(v))).cpu()
 modes=['full','native','mean','isotropic','ridge','isotropic_ridge','normalized_shift','same_task_ridge','same_task_shift'];vectors={};geometry=[];buckets=defaultdict(list)
 for job,c in d['cache'].items():
  k,signal,sign=job;e=ex[k]
  if e['template']!=3:continue
  fold=folds[e['family']];assert k in fold['test'];dc=c['dc'];de=c['de'];z=d['inputs'][k];full=d['vectors'][job]['full'];vv={'full':full,'native':torch.zeros_like(full)};cg={}
  for mode in modes[2:]:
   p=fold['specific'] if mode.startswith('same_task') else fold['shared'];predmode='ridge' if mode=='same_task_ridge' else 'normalized_shift' if mode=='same_task_shift' else mode;pred=predict(p,z,predmode);vv[mode]=T(pred,de);cg[mode]=float((pred-dc).norm()/dc.norm())
  geometry.append(dict(id=k[0],variant=k[1],family=e['family'],signal=signal,sign=sign,context_errors=cg));vectors[job]=vv;buckets[len(e['ids'])].append(job)
 sourcej=json.loads((ART/'bilin18_metric_wording_edit_factorial.json').read_text());competence={fam:sourcej['competence'][fam+'_3'] for fam in families};result=dict(config=cfg,competence=competence,geometry=geometry,interventions=[],summary={},status='running');refs={};replay=[]
 nb=torch.load(ART/'bilin18_agreement_write_subspace_locked_bases.pt',weights_only=False);axis=nb['pca'][0]*torch.sign(nb['pca'][0]@nb['mean']);size=float(axis@nb['mean'])*.0625;rand=torch.randn(1152,generator=torch.Generator().manual_seed(600));rand/=rand.norm()
 def delta(j):
  k,s,sgn=j;v=axis if s=='number' else rand if s=='random' else d['attn'][k[0],1-k[1]]-d['attn'][k]
  if s=='task':v=v/v.norm()
  return v*size*sgn
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save();print('INTERVENTIONS',len(vectors)*len(modes),flush=True)
 for mode in modes:
  for jj in buckets.values():
   for st in range(0,len(jj),16):
    bb=jj[st:st+16];rep=torch.stack([d['cache'][j]['write']+vectors[j][mode] for j in bb]).cuda()
    def hook(m,args,out):y=out.clone();y[:,-1]=rep.to(y);return y
    h=m.register_forward_hook(hook)
    try:z,_=run(model,torch.tensor([ex[j[0]]['ids'] for j in bb],device='cuda'),patch=(5,torch.stack([delta(j) for j in bb])))
    finally:h.remove()
    for i,j in enumerate(bb):
     k,signal,sign=j;e=ex[k];zz=z[i].cpu()
     if mode=='full':refs[j]=zz
     if mode=='native':replay.append(float((zz-d['cache'][j]['logits']).abs().max()))
     rp=refs[j].double().log_softmax(0);lp=zz.double().log_softmax(0);f=vectors[j]['full'];result['interventions'].append(dict(id=k[0],variant=k[1],family=e['family'],signal=signal,sign=sign,mode=mode,margin_error=measure(zz,e['groups'])['margin']-measure(refs[j],e['groups'])['margin'],reference_kl=float((rp.exp()*(rp-lp)).sum()),target_ce_error=float(rp[e['target']]-lp[e['target']]),internal_response_error=float((vectors[j][mode]-f).norm()/f.norm())))
  save();print('COMPLETE',mode,flush=True)
 for fam in families:
  for signal in ['number','task','random']:
   cc=[c for c in result['interventions'] if c['family']==fam and c['signal']==signal];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base]);cd=np.mean([abs(c['target_ce_error']) for c in base])
   for mode in modes:
    vv=[c for c in cc if c['mode']==mode];result['summary'][f'{fam}_{signal}_{mode}']=dict(internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_target_ce_error=float(np.mean([abs(c['target_ce_error']) for c in vv])/cd))
 result['audits']=dict(source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fitted_folds_unchanged=fh==hashlib.sha256(fitted.read_bytes()).hexdigest(),folds=len(folds),training_errors={f:v['training_errors'] for f,v in folds.items()},beta={f:float(v['shared']['beta']) for f,v in folds.items()},shift_gradient_norm={f:v['shared']['shift_gradient_norm'] for f,v in folds.items()},per_fold_training_contexts=32,per_fold_evaluation_contexts=8,training_task_overlap=0,training_wording_overlap=0,same_task_diagnostics_are_exceptions=True,case_count=len(vectors),intervention_count=len(result['interventions']),replay_max_logit_error=max(replay),ridge_normal_error=max(v['shared']['normal_error'] for v in folds.values()),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['margin_error','reference_kl','target_ce_error','internal_response_error']));torch.save(dict(vectors=vectors,folds=folds),ART/f'{STEM}_vectors.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
