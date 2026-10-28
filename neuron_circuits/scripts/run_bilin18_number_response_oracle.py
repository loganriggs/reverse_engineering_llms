"""H595: response-space least-squares oracle in frozen context span."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_number_response_oracle'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_number_bilinear_span.json';saved=torch.load(ART/'bilin18_number_mlp11_response_cross_captures.pt',weights_only=False);all_ex=saved['examples'];inputs=saved['inputs'];writes=saved['writes'];ex={k:e for k,e in all_ex.items() if e['frame']!='preposition'};idx={(e['kind'],e['item'],e['subject_number'],e['attractor_number'],e['frame']):k for k,e in all_ex.items()};pk={k:idx[e['kind'],e['item'],e['subject_number'],e['attractor_number'],'preposition'] for k,e in ex.items()};bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);scale=float(axis@b['mean']);doses=[.0625,.5];ranks=[32]
 cfg=dict(experiment='H595',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),number_basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),doses=doses,ranks=ranks,fit='Freeze the H593 context basis. For each case form columns T(a_j,e); solve float64 SVD least squares against the full target T(dc,e), rcond=1e-10. This is a target-informed per-case oracle, not a learned coefficient predictor. Basis unchanged.',identity='Context-switch response T(dc,e)=Down(Left(dc)*Right(e)+Left(e)*Right(dc)). Quadratic-in-e term cancels exactly.',design='Compare full target, unchanged context, Euclidean context projection and response-optimal context-span oracle. Evaluate live downstream fidelity at both signs/doses across all H593 cells. Audit conditioning, coefficient norms, normal equations and direct coefficient replay.',scope='Oracle sees each full local response target and actual incoming perturbation. It minimizes local Euclidean response error only, not output KL or answer loss; success is not generalization. Counterfactual coefficient norms and conditioning are reported.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));priorbases=torch.load(ART/'bilin18_number_bilinear_subspace_locked_bases.pt',weights_only=False);train=priorbases['training_ids'];A=priorbases['context'];B=priorbases['incoming'];cs=priorbases['context_singular_values'];es=priorbases['incoming_singular_values'];random={};basispath=ART/f'{STEM}_locked_bases.pt';torch.save(dict(context=A,incoming=B,context_singular_values=cs,incoming_singular_values=es,random=random,training_ids=train),basispath);bh=hashlib.sha256(basispath.read_bytes()).hexdigest();print('LOCKED BASES',bh,flush=True)
 model=load_full_model('cuda');assert not model.config.gated
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 mlp=model.blocks[11].mlp
 def T(c,e):
  c=c.cuda();e=e.cuda();return mlp.Down(mlp.Left(c)*mlp.Right(e)+mlp.Left(e)*mlp.Right(c)).cpu()
 vecs={};buckets=defaultdict(list);identityerrors=[];normerrors=[];modes=['full','native','context_32','oracle_32'];oracle_geometry=[];coefficients={}
 for k,e in ex.items():
  dc=inputs[pk[k],0.]-inputs[k,0.]
  for dose in [sgn*d for d in doses for sgn in (-1,1)]:
   de=inputs[k,dose]-inputs[k,0.];full=T(dc,de);vv={'full':full,'native':torch.zeros_like(full)}
   projected=A[:32].T@(A[:32]@dc);vv['context_32']=T(projected,de)
   matrix=T(A[:32],de).T.double();target=full.double();U,S,Vh=torch.linalg.svd(matrix,full_matrices=False);keep=S>S[0]*1e-10;coef=Vh[keep].T@((U[:,keep].T@target)/S[keep]);pred=matrix@coef;res=pred-target;vv['oracle_32']=pred.float();coefficients[k,dose]=coef
   mapped=T((A[:32].double().T@coef).float(),de).double();ortho=float((matrix.T@res).norm()/(matrix.norm()*target.norm()));oracle_geometry.append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=dose,rank=int(keep.sum()),condition=float(S[0]/S[-1]),normal_equation_relative_error=ortho,coefficient_norm_ratio=float(coef.norm()/dc.double().norm()),projection_response_error=float((vv['context_32'].double()-target).norm()/target.norm()),oracle_response_error=float(res.norm()/target.norm()),coefficient_replay_relative_error=float((mapped-pred).norm()/target.norm())))
   actual=saved['hybrids'][k,dose,'preposition_recipient']-saved['hybrids'][k,dose,'recipient_recipient'];identityerrors.append(float((actual-full).norm()/full.norm()))
   for mode,v in vv.items():vecs[k,dose,mode]=v
   buckets[len(e['ids'])].append((k,dose))
 print('LOCKED',len(vecs),'interventions',flush=True);refs={};result=dict(config=cfg,basis_sha256=bh,discovery_energy={str(r):dict(context=float(cs[:r].square().sum()/cs.square().sum()),incoming=float(es[:r].square().sum()/es.square().sum())) for r in ranks},oracle_geometry=oracle_geometry,interventions=[],summary={},status='running')
 for mode in modes:
  for length,jj in buckets.items():
   for st in range(0,len(jj),24):
    bb=jj[st:st+24];don=torch.stack([writes[k,0.]+saved['hybrids'][k,dose,'recipient_recipient']+vecs[k,dose,mode] for k,dose in bb]).cuda()
    def h(m,args,out):out=out.clone();out[:,-1]=don.to(out);return out
    handle=mlp.register_forward_hook(h)
    try:zz,_=run(model,torch.tensor([ex[k]['ids'] for k,d in bb],device='cuda'),patch=(5,torch.stack([axis*scale*dose for k,dose in bb])))
    finally:handle.remove()
    for ix,(k,dose) in enumerate(bb):
     z=zz[ix].cpu();e=ex[k];gs=e['groups'];ans=gs[e['subject_number']]
     if mode in ('full','native'):refs[k,dose,mode]=z
     ref=refs[k,dose,'full'];lp=z.double().log_softmax(0);rp=ref.double().log_softmax(0);margin=measure(z,gs)['margin'];rm=measure(ref,gs)['margin'];f=vecs[k,dose,'full'];result['interventions'].append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=dose,mode=mode,margin=margin,margin_error=margin-rm,reference_kl=float((rp.exp()*(rp-lp)).sum()),group_nll_error=float(torch.logsumexp(rp[ans],0)-torch.logsumexp(lp[ans],0)),internal_response_error=float((vecs[k,dose,mode]-f).norm()/f.norm())))
  print('COMPLETE',mode,flush=True)
 for kind in ('regular','irregular'):
  for frame in ('fronted_preposition','relative_clause'):
   for dose in doses:
    cc=[c for c in result['interventions'] if c['kind']==kind and c['frame']==frame and abs(c['dose'])==dose];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base])
    for mode in modes:
     vv=[c for c in cc if c['mode']==mode];result['summary'][f'{kind}_{frame}_{dose}_{mode}']=dict(internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),group_nll_abs_error=float(np.mean([abs(c['group_nll_error']) for c in vv])))
 prior=json.loads((ART/'bilin18_number_mlp11_response_cross.json').read_text());old={(c['id'],c['dose'],c['mode']):c for c in prior['interventions']};nat={}
 for stem in ('bilin18_number_subject_attractor','bilin18_number_fronted_attractor'):nat.update(torch.load(ART/f'{stem}_cache.pt',weights_only=False)['native'])
 replay=[]
 for c in result['interventions']:
  if c['mode'] not in ('full','native'):continue
  nm=measure(nat[c['id']],ex[c['id']]['groups'])['margin'];om='preposition_recipient' if c['mode']=='full' else 'recipient_recipient';replay.append(abs(nm-c['margin']-old[c['id'],c['dose'],om]['plural_shift']))
 result['audits']=dict(oracle_normal_equation_max_relative_error=max(g['normal_equation_relative_error'] for g in oracle_geometry),oracle_coefficient_replay_max_relative_error=max(g['coefficient_replay_relative_error'] for g in oracle_geometry),oracle_worse_than_projection_count=sum(g['oracle_response_error']>g['projection_response_error']+1e-6 for g in oracle_geometry),context_switch_identity_max_relative_error=max(identityerrors),matched_input_norm_max_relative_error=max(normerrors,default=0.),reference_replay_max_margin_error=max(replay),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ('margin_error','reference_kl','internal_response_error')),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=bh==hashlib.sha256(basispath.read_bytes()).hexdigest(),number_basis_unchanged=cfg['number_basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(dict(vectors=vecs,coefficients=coefficients,examples=ex,oracle_geometry=oracle_geometry),ART/f'{STEM}_oracle.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()
