"""H588: identical frozen-axis perturbations separate write size from downstream response."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_number_fixed_dose'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_number_fronted_attractor.json';bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);scale=float(axis@b['mean']);directions={'number':axis,'nuisance':b['nuisance'][0],**{f'random{s}':v[0] for s,v in b['random'].items()}};ex={};native={}
 for stem in ('bilin18_number_subject_attractor','bilin18_number_fronted_attractor'):
  saved=torch.load(ART/f'{stem}_cache.pt',weights_only=False);assert not set(ex)&set(saved['examples']);ex.update(saved['examples']);native.update(saved['native'])
 doses=[.0625,.25,.5,1.];cfg=dict(experiment='H588',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),layer=5,site=-1,discovery_amplitude=scale,doses=doses,directions=list(directions),design='Identical signed perturbations at Attention5 final write for all 192 H586/H587 prompts. Unit direction times fixed amplitude (original discovery mean projected on oriented PC1) times signed dose. Number axis, nuisance axis and five seeded random axes have equal norms. Zero once per prompt. No prompt-dependent rescaling.',metrics='Native-to-intervened KL, agreement margin/plural shift, intended-group NLL. Central finite-difference response per discovery-scale unit at four doses; nonlinear curvature and monotonicity. Native/full-swap data remain distinct.',scope='All existing frames and number/type cells retained. Frame changes syntax, position, lexical surroundings and native state; this identifies contextual downstream sensitivity, not a unique gating component.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 jobs=defaultdict(list)
 for k,e in ex.items():
  jobs[len(e['ids'])].append((k,'zero',0.))
  for dn in directions:
   for dose in doses:
    for sign in (-1,1):jobs[len(e['ids'])].append((k,dn,sign*dose))
 print('LOCKED',sum(map(len,jobs.values())),'scale',scale,flush=True);result=dict(config=cfg,interventions=[],responses=[],summary={},status='running');finished=0
 for T,jj in jobs.items():
  for st in range(0,len(jj),24):
   bb=jj[st:st+24];delta=torch.stack([directions.get(dn,axis)*scale*dose for k,dn,dose in bb]);z,_=run(model,torch.tensor([ex[k]['ids'] for k,dn,dose in bb],device='cuda'),patch=(5,delta))
   for ix,(k,dn,dose) in enumerate(bb):
    e=ex[k];nz=native[k].to(z);lp=z[ix].log_softmax(0);np_=nz.log_softmax(0);nm=measure(nz,e['groups'])['margin'];margin=measure(z[ix],e['groups'])['margin'];ans=e['groups'][e['subject_number']];result['interventions'].append(dict(id=k,kind=e['kind'],frame=e['frame'],subject_number=e['subject_number'],attractor_number=e['attractor_number'],direction=dn,dose=dose,plural_shift=nm-margin,kl=float((np_.exp()*(np_-lp)).sum()),group_nll_delta=float(torch.logsumexp(np_[ans],0)-torch.logsumexp(lp[ans],0))))
  finished+=len(jj);print('PROGRESS',finished,flush=True)
 lookup={(c['id'],c['direction'],c['dose']):c for c in result['interventions']}
 for k,e in ex.items():
  for dn in directions:
   for dose in doses:
    p=lookup[k,dn,dose];n=lookup[k,dn,-dose];result['responses'].append(dict(id=k,kind=e['kind'],frame=e['frame'],subject_number=e['subject_number'],attractor_number=e['attractor_number'],direction=dn,dose=dose,central_plural_response=(p['plural_shift']-n['plural_shift'])/(2*dose),even_response=(p['plural_shift']+n['plural_shift'])/2,mean_kl=(p['kl']+n['kl'])/2,mean_group_nll_delta=(p['group_nll_delta']+n['group_nll_delta'])/2))
 for kind in ('regular','irregular'):
  for frame in ('preposition','relative_clause','fronted_preposition'):
   for dn in directions:
    for dose in doses:
     cc=[c for c in result['responses'] if (c['kind'],c['frame'],c['direction'],c['dose'])==(kind,frame,dn,dose)];result['summary'][f'{kind}_{frame}_{dn}_{dose}']=dict(n=len(cc),mean_central_plural_response=float(np.mean([c['central_plural_response'] for c in cc])),median_central_plural_response=float(np.median([c['central_plural_response'] for c in cc])),positive_fraction=float(np.mean([c['central_plural_response']>0 for c in cc])),mean_absolute_even_response=float(np.mean([abs(c['even_response']) for c in cc])),mean_kl=float(np.mean([c['mean_kl'] for c in cc])),mean_group_nll_delta=float(np.mean([c['mean_group_nll_delta'] for c in cc])))
 result['audits']=dict(prompt_count=len(ex),zero_margin_max_error=max(abs(c['plural_shift']) for c in result['interventions'] if c['direction']=='zero'),unit_norm_max_error=max(abs(float(v.norm())-1) for v in directions.values()),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['plural_shift','kl','group_nll_delta']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=cfg['basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(result['audits'],flush=True)
if __name__=='__main__':main()
