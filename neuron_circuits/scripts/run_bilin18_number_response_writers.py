"""H590: exact response accounting, discovery-ranked writer clamps and lexical validation."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_number_response_writers'
OPS=[('mlp',5)]+[(kind,l) for l in range(6,18) for kind in ('attn','mlp')]
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_number_terminal_response_cross.json';bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);scale=float(axis@b['mean']);ex={}
 for stem in ('bilin18_number_subject_attractor','bilin18_number_fronted_attractor'):ex.update(torch.load(ART/f'{stem}_cache.pt',weights_only=False)['examples'])
 doses=[.0625,.5];cfg=dict(experiment='H590',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),doses=doses,operations=OPS,design='Capture native and signed fixed-number perturbation writes after Attention5. Reconstruct final central residual response from injected skip path plus all 25 scaled downstream attention/MLP write responses. Map terms through recipient-matched prepositional native terminal-readout gradient. Accounting is descriptive.',selection='At small dose on regular nouns only, rank components by most negative mean mapped contribution difference: fronted minus unfronted. Lock top three before clamps. Test single selected clamps, joint clamp, all downstream response clamps and unchanged control on all frames/types at both doses.',clamp='Overwrite a selected final-position module output with its same-prompt unperturbed value while keeping the original Attention5 perturbation and all other computation live. All-clamp should leave only the scaled direct residual path.',scope='Irregular rows validate discovery selection on an already-explored panel. Terminal gradient is a local comparison chart; finite-response linearization audited. No unique additive causal mediation claim.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for k,e in ex.items():buckets[len(e['ids'])].append(k)
 captures={};native={};final={}
 with torch.no_grad():
  for T,kk in buckets.items():
   for st in range(0,len(kk),16):
    bb=kk[st:st+16];ids=torch.tensor([ex[k]['ids'] for k in bb],device='cuda')
    for dose in [0.]+[sgn*d for d in doses for sgn in (-1,1)]:
     cap={};handles=[]
     for kind,l in OPS:
      def wh(m,args,out,kind=kind,l=l):cap[kind,l]=(out[0] if kind=='attn' else out)[:,-1].cpu()
      handles.append(getattr(model.blocks[l],kind).register_forward_hook(wh))
     try:z,cc=run(model,ids,capture=True,patch=None if dose==0 else (5,(axis*scale*dose).expand(len(bb),-1)))
     finally:
      for h in handles:h.remove()
     for ix,k in enumerate(bb):captures[k,dose]={n:v[ix].clone() for n,v in cap.items()};final[k,dose]=cc['last'][ix].clone()
     if dose==0:
      for ix,k in enumerate(bb):native[k]=z[ix].cpu()
 gains={}
 for l in range(5,18):
  gain=1.
  for future in range(l+1,18):gain*=float(model.blocks[future].lambdas[0])
  gains[l]=gain
 idx={(e['kind'],e['item'],e['subject_number'],e['attractor_number'],e['frame']):k for k,e in ex.items()};grad={}
 for k,e in ex.items():
  if e['frame']!='preposition':continue
  x=final[k,0.].cuda().detach().requires_grad_(True);z=30*torch.tanh(model.lm_head(F.rms_norm(x,(x.numel(),))).float()/30);score=torch.logsumexp(z[e['groups'][1]],0)-torch.logsumexp(z[e['groups'][0]],0);grad[k]=torch.autograd.grad(score,x)[0].detach().cpu()
 result=dict(config=cfg,accounting=[],selection=[],interventions=[],responses=[],summary={},status='running');errs=[];normerr=[];prior=json.loads((ART/'bilin18_number_terminal_response_cross.json').read_text());priorresp={(c['id'],c['donor_frame'],c['dose']):c for c in prior['responses']};linerrors=[]
 for k,e in ex.items():
  pk=idx[e['kind'],e['item'],e['subject_number'],e['attractor_number'],'preposition'];g=grad[pk]
  for dose in doses:
   response=(final[k,dose]-final[k,-dose])/(2*dose);terms={'direct':gains[5]*axis*scale}
   for kind,l in OPS:terms[f'{kind}_{l}']=gains[l]*(captures[k,dose][kind,l]-captures[k,-dose][kind,l])/(2*dose)
   err=(sum(terms.values())-response);errs.append(float(err.norm()/response.norm()));normerr.append(float(err.abs().max()));linerrors.append(dict(dose=dose,error=float(g@response)-priorresp[pk,e['frame'],dose]['central_plural_response']))
   for name,v in terms.items():result['accounting'].append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=dose,component=name,mapped_response=float(g@v),response_norm=float(v.norm())))
 scores={}
 for kind,l in OPS:
  mode=f'{kind}_{l}';cc=[c for c in result['accounting'] if c['kind']=='regular' and c['dose']==.0625 and c['component']==mode];scores[mode]=float(np.mean([c['mapped_response'] for c in cc if c['frame']=='fronted_preposition'])-np.mean([c['mapped_response'] for c in cc if c['frame']=='preposition']))
 selected=sorted(scores,key=scores.get)[:3];result['selection']=selected;result['selection_scores']=scores;lock=ART/f'{STEM}_locked_selection.json';lock.write_text(json.dumps(dict(selected=selected,scores=scores),indent=2));print('LOCKED SELECTION',selected,flush=True)
 modes=['unchanged']+selected+['joint','all'];jobs=defaultdict(list)
 for k,e in ex.items():
  for mode in modes:
   for dose in doses:
    for sign in (-1,1):jobs[len(e['ids']),mode,sign*dose].append(k)
 allerrors=[]
 with torch.no_grad():
  for (T,mode,dose),kk in jobs.items():
   for st in range(0,len(kk),24):
    bb=kk[st:st+24];handles=[];chosen=[] if mode=='unchanged' else OPS if mode=='all' else [(m.split('_')[0],int(m.split('_')[1])) for m in selected] if mode=='joint' else [(mode.split('_')[0],int(mode.split('_')[1]))]
    for kind,l in chosen:
     don=torch.stack([captures[k,0.][kind,l] for k in bb]).cuda()
     if kind=='attn':
      def wh(m,args,out,don=don):y,bus=out;y=y.clone();y[:,-1]=don.to(y);return y,bus
     else:
      def wh(m,args,out,don=don):y=out.clone();y[:,-1]=don.to(y);return y
     handles.append(getattr(model.blocks[l],kind).register_forward_hook(wh))
    try:z,cc=run(model,torch.tensor([ex[k]['ids'] for k in bb],device='cuda'),capture=mode=='all',patch=(5,(axis*scale*dose).expand(len(bb),-1)))
    finally:
     for h in handles:h.remove()
    for ix,k in enumerate(bb):
     e=ex[k];nz=native[k].to(z);lp=z[ix].log_softmax(0);np_=nz.log_softmax(0);ans=e['groups'][e['subject_number']];result['interventions'].append(dict(id=k,kind=e['kind'],frame=e['frame'],mode=mode,dose=dose,plural_shift=measure(nz,e['groups'])['margin']-measure(z[ix],e['groups'])['margin'],kl=float((np_.exp()*(np_-lp)).sum()),group_nll_delta=float(torch.logsumexp(np_[ans],0)-torch.logsumexp(lp[ans],0))))
     if mode=='all':
      expected=final[k,0.]+gains[5]*axis*scale*dose;allerrors.append(float((cc['last'][ix]-expected).norm()/(gains[5]*axis*scale*dose).norm()))
 lookup={(c['id'],c['mode'],c['dose']):c for c in result['interventions']}
 for k,e in ex.items():
  for mode in modes:
   for dose in doses:result['responses'].append(dict(id=k,kind=e['kind'],frame=e['frame'],mode=mode,dose=dose,central_plural_response=(lookup[k,mode,dose]['plural_shift']-lookup[k,mode,-dose]['plural_shift'])/(2*dose)))
 for kind in ('regular','irregular'):
  for frame in ('preposition','relative_clause','fronted_preposition'):
   for mode in modes:
    for dose in doses:
     cc=[c['central_plural_response'] for c in result['responses'] if (c['kind'],c['frame'],c['mode'],c['dose'])==(kind,frame,mode,dose)];result['summary'][f'{kind}_{frame}_{mode}_{dose}']=dict(mean_response=float(np.mean(cc)),positive_fraction=float(np.mean(np.array(cc)>0)))
 old=json.loads((ART/'bilin18_number_fixed_dose.json').read_text());ol={(c['id'],c['dose']):c for c in old['interventions'] if c['direction']=='number'};result['audits']=dict(response_reconstruction_max_relative_error=max(errs),response_reconstruction_max_coordinate_error=max(normerr),all_clamp_direct_path_max_relative_error=max(allerrors),unchanged_H588_max_margin_error=max(abs(c['plural_shift']-ol[c['id'],c['dose']]['plural_shift']) for c in result['interventions'] if c['mode']=='unchanged'),linearization_mean_abs_error_by_dose={str(d):float(np.mean([abs(c['error']) for c in linerrors if c['dose']==d])) for d in doses},finite=all(np.isfinite(c['plural_shift']) and np.isfinite(c['kl']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=cfg['basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest(),selection_unchanged=selected==json.loads(lock.read_text())['selected']);result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(dict(captures=captures,final=final,examples=ex,gains=gains),ART/f'{STEM}_captures.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()
