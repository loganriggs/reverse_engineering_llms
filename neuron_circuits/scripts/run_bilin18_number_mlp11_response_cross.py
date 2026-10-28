"""H591: cross MLP11 native context and incoming normalized-input response."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_number_mlp11_response_cross'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_number_response_writers.json';bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);scale=float(axis@b['mean']);ex={}
 for stem in ('bilin18_number_subject_attractor','bilin18_number_fronted_attractor'):ex.update(torch.load(ART/f'{stem}_cache.pt',weights_only=False)['examples'])
 doses=[.0625,.5];cfg=dict(experiment='H591',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),doses=doses,design='At MLP11 normalized input, cross recipient versus matched unfronted native context z with recipient versus unfronted input response e caused by fixed signed Attention5 perturbation. Replace only MLP11 response around recipient native output; all other downstream computation live. Four crossings per recipient/frame, both doses/signs. Prepositional cells are duplicate/self controls.',identity='B(z+e)-B(z)=Down(Left(z)*Right(e)+Left(e)*Right(z)+Left(e)*Right(e)). Audit direct native output differences and separate linear/quadratic terms.',scope='Counterfactual post-normalization context/response crossings need not be natural states. Distinguishes conditional bilinear response from incoming normalized-input response, not whole-layer normalization or unique semantic features.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda');assert not model.config.gated
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 mlp=model.blocks[11].mlp;buckets=defaultdict(list)
 for k,e in ex.items():buckets[len(e['ids'])].append(k)
 inputs={};writes={};native={}
 for T,kk in buckets.items():
  for st in range(0,len(kk),24):
   bb=kk[st:st+24];ids=torch.tensor([ex[k]['ids'] for k in bb],device='cuda')
   for dose in [0.]+[sgn*d for d in doses for sgn in (-1,1)]:
    cap={}
    def pre(m,args):cap['z']=args[0][:,-1].cpu()
    def post(m,args,out):cap['w']=out[:,-1].cpu()
    handles=[mlp.register_forward_pre_hook(pre),mlp.register_forward_hook(post)]
    try:zz,_=run(model,ids,patch=None if dose==0 else (5,(axis*scale*dose).expand(len(bb),-1)))
    finally:
     for h in handles:h.remove()
    for ix,k in enumerate(bb):inputs[k,dose]=cap['z'][ix].clone();writes[k,dose]=cap['w'][ix].clone()
    if dose==0:
     for ix,k in enumerate(bb):native[k]=zz[ix].cpu()
 idx={(e['kind'],e['item'],e['subject_number'],e['attractor_number'],e['frame']):k for k,e in ex.items()};jobs=defaultdict(list);hybrids={};errors=[];result=dict(config=cfg,geometry=[],interventions=[],responses=[],summary={},status='running')
 for k,e in ex.items():
  pk=idx[e['kind'],e['item'],e['subject_number'],e['attractor_number'],'preposition']
  for dose in [sgn*d for d in doses for sgn in (-1,1)]:
   for context in ('recipient','preposition'):
    ck=k if context=='recipient' else pk;z=inputs[ck,0.].cuda();lz=mlp.Left(z);rz=mlp.Right(z)
    for incoming in ('recipient','preposition'):
     ik=k if incoming=='recipient' else pk;de=(inputs[ik,dose]-inputs[ik,0.]).cuda();le=mlp.Left(de);re=mlp.Right(de);linear=mlp.Down(lz*re+le*rz);quad=mlp.Down(le*re);response=linear+quad;mode=context+'_'+incoming;hybrids[k,dose,mode]=response.cpu();jobs[len(e['ids']),dose,mode].append(k)
     if mode=='recipient_recipient':
      actual=(writes[k,dose]-writes[k,0.]).cuda();errors.append(float((response-actual).norm()/actual.norm()));result['geometry'].append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=dose,input_response_norm=float(de.norm()),response_norm=float(response.norm()),quadratic_relative_norm=float(quad.norm()/response.norm())))
 print('LOCKED',sum(map(len,jobs.values())),flush=True)
 for (T,dose,mode),kk in jobs.items():
  for st in range(0,len(kk),24):
   bb=kk[st:st+24];don=torch.stack([writes[k,0.]+hybrids[k,dose,mode] for k in bb]).cuda()
   def wh(m,args,out):y=out.clone();y[:,-1]=don.to(y);return y
   h=mlp.register_forward_hook(wh)
   try:zz,_=run(model,torch.tensor([ex[k]['ids'] for k in bb],device='cuda'),patch=(5,(axis*scale*dose).expand(len(bb),-1)))
   finally:h.remove()
   for ix,k in enumerate(bb):
    e=ex[k];nz=native[k].to(zz);lp=zz[ix].log_softmax(0);np_=nz.log_softmax(0);ans=e['groups'][e['subject_number']];result['interventions'].append(dict(id=k,kind=e['kind'],frame=e['frame'],mode=mode,dose=dose,plural_shift=measure(nz,e['groups'])['margin']-measure(zz[ix],e['groups'])['margin'],kl=float((np_.exp()*(np_-lp)).sum()),group_nll_delta=float(torch.logsumexp(np_[ans],0)-torch.logsumexp(lp[ans],0))))
 lookup={(c['id'],c['mode'],c['dose']):c for c in result['interventions']};modes=['recipient_recipient','preposition_recipient','recipient_preposition','preposition_preposition']
 for k,e in ex.items():
  for mode in modes:
   for dose in doses:result['responses'].append(dict(id=k,kind=e['kind'],frame=e['frame'],mode=mode,dose=dose,central_plural_response=(lookup[k,mode,dose]['plural_shift']-lookup[k,mode,-dose]['plural_shift'])/(2*dose)))
 for kind in ('regular','irregular'):
  for frame in ('preposition','relative_clause','fronted_preposition'):
   for mode in modes:
    for dose in doses:
     cc=[c['central_plural_response'] for c in result['responses'] if (c['kind'],c['frame'],c['mode'],c['dose'])==(kind,frame,mode,dose)];result['summary'][f'{kind}_{frame}_{mode}_{dose}']=dict(mean_response=float(np.mean(cc)),positive_fraction=float(np.mean(np.array(cc)>0)))
 prior=json.loads((ART/'bilin18_number_fixed_dose.json').read_text());old={(c['id'],c['dose']):c for c in prior['interventions'] if c['direction']=='number'};result['audits']=dict(bilinear_response_reconstruction_max_relative_error=max(errors),native_response_replay_max_margin_error=max(abs(c['plural_shift']-old[c['id'],c['dose']]['plural_shift']) for c in result['interventions'] if c['mode']=='recipient_recipient'),preposition_duplicate_max_margin_error=max(abs(c['plural_shift']-lookup[c['id'],'recipient_recipient',c['dose']]['plural_shift']) for c in result['interventions'] if c['frame']=='preposition'),finite=all(np.isfinite(c['plural_shift']) and np.isfinite(c['kl']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=cfg['basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(dict(inputs=inputs,writes=writes,hybrids=hybrids,examples=ex),ART/f'{STEM}_captures.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()
