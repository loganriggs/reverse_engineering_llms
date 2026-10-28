"""H634 discovery-only response subspaces with context and input-direction holdout."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_cross_task_response_subspaces'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();cp=ART/'bilin18_cross_task_response_panel_captures.pt';mp=ART/'bilin18_cross_task_response_panel_manifest.json';jp=ART/'bilin18_cross_task_response_panel.json';paths={'source':cp,'manifest':mp,'source_results':jp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=json.loads(mp.read_text());source=torch.load(cp,weights_only=False);old=json.loads(jp.read_text());assert old['status']=='complete';names=list(source['responses']);train_names=['original','span0','span1','orthogonal0','orthogonal1'];held_names=[n for n in names if n not in train_names];fit=[i for i,e in enumerate(rows) if e['split']=='fit'];families=sorted({e['family'] for e in rows});ranks={'global':[1,4,16,32,100],**{f:[1,4,16,20] for f in families}}
 cfg=dict(experiment='H634',hashes=hashes,fit_directions=train_names,heldout_directions=held_names,ranks=ranks,design='Discovery-only weighted SVD of output derivatives. Shared pooled discovery probability metric. Fit20 contexts x5 directions globally,4x5 per task. Evaluate receiver-weighted optimal projections on all responses, report heldout contexts and heldout input directions separately.',limitations='Oracle coefficients fitted to each test response: representational upper bound, not prediction. Direction partition declared after H633 observations, so retrospective holdout, not a prospective blind replication. Initial fixed ranks1/4/16/32 and task20; global100 added as an exploratory full-discovery-span ceiling after inspecting initial results. No best-rank selection.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));Z=source['native_logits'].double().cuda();P=Z.softmax(-1);pfit=P[fit].mean(0);root=pfit.sqrt();G={n:g.cuda() for n,g in source['responses'].items()};bases={};spectra={};ortherr=0.
 for group in ranks:
  ii=fit if group=='global' else [i for i in fit if rows[i]['family']==group];M=torch.stack([G[n][i] for i in ii for n in train_names]);M=M-(M*pfit).sum(-1,keepdim=True);_,sv,V=torch.linalg.svd(M*root,full_matrices=False);B=V[:max(ranks[group])]/root;bases[group]=B;spectra[group]=dict(singular_values=sv.cpu().tolist(),training_rows=len(M));ortherr=max(ortherr,float(((B*pfit)@B.T-torch.eye(len(B),device='cuda')).abs().max()));print('FIT',group,flush=True)
 records=[];finite=[];max_normal_equation=0.;max_nesting_violation=0.
 for i,e in enumerate(rows):
  p=P[i];actual=torch.stack([G[n][i] for n in names]);actual-= (actual*p).sum(-1,keepdim=True);den=(actual.square()*p).sum(-1);prior={}
  for group in ['global',e['family']]:
   prev=None
   for rank in ranks[group]:
    B=bases[group][:rank];B=B-(B*p).sum(-1,keepdim=True);gram=(B*p)@B.T;rhs=(actual*p)@B.T;coef=torch.linalg.solve(gram,rhs.T).T;pred=coef@B;res=actual-pred;err=(res.square()*p).sum(-1)/den;normal=((res*p)@B.T).abs().max();max_normal_equation=max(max_normal_equation,float(normal))
    if prev is not None:max_nesting_violation=max(max_nesting_violation,float((err-prev).max()))
    prev=err
    for q,name in enumerate(names):records.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],probe=name,direction_split='fit' if name in train_names else 'heldout',group='global' if group=='global' else 'task',rank=rank,relative_error=float(err[q].clamp_min(0).sqrt()),energy_captured=float(1-err[q])))
    for alpha,z in source['finite_logits'].items():
     truth=z[i].double().cuda().log_softmax(0);lp=(Z[i]+alpha*pred[names.index('original')]).log_softmax(0);finite.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],group='global' if group=='global' else 'task',rank=rank,alpha=alpha,kl=float((truth.exp()*(truth-lp)).sum())))
  if i%10==0:print('PROJECTED',i,flush=True)
 summary={}
 for f in families:
  for form in [0,1]:
   for ds in ['fit','heldout']:
    for group in ['global','task']:
     for rank in ranks['global' if group=='global' else f]:
      rr=[r for r in records if r['family']==f and r['form']==form and r['split']=='heldout' and r['direction_split']==ds and r['group']==group and r['rank']==rank];summary[f'{f}_{form}_{ds}_{group}_{rank}']=dict(n=len(rr),mean_error=float(np.mean([r['relative_error'] for r in rr])),median_error=float(np.median([r['relative_error'] for r in rr])),mean_energy=float(np.mean([r['energy_captured'] for r in rr])))
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},finite=all(np.isfinite(r['relative_error']) for r in records),weighted_basis_orthogonality_max=ortherr,normal_equation_max=max_normal_equation,nesting_violation_max=max_nesting_violation,fit_contexts=len(fit),fit_directions=len(train_names),heldout_directions=len(held_names),new_forwards=0)
 torch.save(dict(bases={k:v.cpu() for k,v in bases.items()},discovery_probability=pfit.cpu()),ART/f'{STEM}_locked.pt');result=dict(status='complete',config=cfg,records=records,finite_kl=finite,summary=summary,spectra=spectra,audits=audits,elapsed_seconds=time.time()-start);(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
