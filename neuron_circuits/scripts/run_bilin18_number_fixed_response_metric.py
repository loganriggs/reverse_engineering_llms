"""H598: frozen response-weighted context projection without per-case target responses."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_number_fixed_response_metric'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_number_fresh_transfer.json';prior=json.loads(src.read_text());fresh=torch.load(ART/'bilin18_number_fresh_transfer_captures.pt',weights_only=False);all_ex=fresh['examples'];inputs=fresh['inputs'];writes=fresh['writes'];captured_logits=fresh['logits'];saved={'hybrids':fresh['hybrids']};ex={k:e for k,e in all_ex.items() if not e['frame'].startswith('preposition')};idx={(e['kind'],e['item'],e['subject_number'],e['attractor_number'],e['frame']):k for k,e in all_ex.items()};pk={k:idx[e['kind'],e['item'],e['subject_number'],e['attractor_number'],'preposition_'+e['wording']] for k,e in ex.items()};competence=prior['competence'];bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);scale=float(axis@b['mean']);doses=[.0625];frozen=torch.load(ART/'bilin18_number_context_coverage_locked_bases.pt',weights_only=False);A=frozen['spaces']['new64'];spaces={'new64':A};containment=0.;basispath=ART/f'{STEM}_metric.pt';old=torch.load(ART/'bilin18_number_mlp11_response_cross_captures.pt',weights_only=False);training=sorted(k for k,e in old['examples'].items() if e['kind']=='regular' and e['frame']!='preposition');assert len(training)==64 and not set(training)&set(all_ex)
 cfg=dict(experiment='H598',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),number_basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),doses=doses,design='Freeze new64 context basis. Build a fixed response-weighted projection from the 64 old regular discovery positive small-dose input responses only. K=(A G A^T)^-1 A G where G=mean J_e^T J_e and J_e dc=T(dc,e). At evaluation coefficients are K dc; no new-case e or full target used for coefficients. Use actual new incoming e to execute the same bilinear map.',comparison='Full, unchanged, Euclidean projection, per-case oracle, fixed metric. Evaluate all H597 fresh cases. H597 outcomes motivated this follow-up, so this is adaptive evaluation, not a second independent confirmation.',scope='Fixed linear coefficient map, not an upstream context generator. Uses original model weights and discovery input responses to learn a metric; oracle remains diagnostic only. Float64 KL and linear solve.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda');assert not model.config.gated
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 mlp=model.blocks[11].mlp;ac=A.cuda();cross=torch.zeros(64,A.shape[1],dtype=torch.float64);la=mlp.Left(ac);ra=mlp.Right(ac)
 for k in training:
  e=(old['inputs'][k,.0625]-old['inputs'][k,0.]).cuda();le=mlp.Left(e);re=mlp.Right(e);M=mlp.Down(la*re+ra*le);back=M@mlp.Down.weight;ag=(back*re)@mlp.Left.weight+(back*le)@mlp.Right.weight;cross+=ag.cpu().double()/len(training)
 H=cross@A.double().T;K=torch.linalg.solve(H,cross).float();torch.save(dict(K=K,basis=A,cross=cross,H=H,training_ids=training),basispath);bh=hashlib.sha256(basispath.read_bytes()).hexdigest();print('LOCKED METRIC',bh,flush=True)
 def T(c,e):
  c=c.cuda();e=e.cuda();return mlp.Down(mlp.Left(c)*mlp.Right(e)+mlp.Left(e)*mlp.Right(c)).cpu()
 vecs={};buckets=defaultdict(list);identityerrors=[];normerrors=[];modes=['full','native','new64_projection','new64_oracle','metric_new64'];oracle_geometry=[];coefficients={}
 for k,e in ex.items():
  dc=inputs[pk[k],0.]-inputs[k,0.]
  for dose in [sgn*d for d in doses for sgn in (-1,1)]:
   de=inputs[k,dose]-inputs[k,0.];full=T(dc,de);vv={'full':full,'native':torch.zeros_like(full)}
   for space,A in spaces.items():
    projected=A.T@(A@dc);vv[space+'_projection']=T(projected,de)
    matrix=T(A,de).T.double();target=full.double();U,S,Vh=torch.linalg.svd(matrix,full_matrices=False);keep=S>S[0]*1e-10;coef=Vh[keep].T@((U[:,keep].T@target)/S[keep]);pred=matrix@coef;res=pred-target;vv[space+'_oracle']=pred.float();coefficients[k,dose,space]=coef
    mapped=T((A.double().T@coef).float(),de).double();ortho=float((matrix.T@res).norm()/(matrix.norm()*target.norm()));oracle_geometry.append(dict(id=k,kind=e['kind'],frame=e['frame'],dose=dose,space=space,rank=int(keep.sum()),condition=float(S[0]/S[-1]),normal_equation_relative_error=ortho,coefficient_norm_ratio=float(coef.norm()/dc.double().norm()),projection_response_error=float((vv[space+'_projection'].double()-target).norm()/target.norm()),oracle_response_error=float(res.norm()/target.norm()),coefficient_replay_relative_error=float((mapped-pred).norm()/target.norm())))
   vv['metric_new64']=T(A.T@(K@dc),de)
   actual=saved['hybrids'][k,dose,'preposition_recipient']-saved['hybrids'][k,dose,'recipient_recipient'];identityerrors.append(float((actual-full).norm()/full.norm()))
   for mode,v in vv.items():vecs[k,dose,mode]=v
   buckets[len(e['ids'])].append((k,dose))
 print('LOCKED',len(vecs),'interventions',flush=True);refs={};result=dict(config=cfg,competence=competence,basis_sha256=bh,oracle_geometry=oracle_geometry,interventions=[],summary={},status='running')
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
  for frame in ('fronted_old','relative_old','fronted_new','relative_new'):
   for dose in doses:
    cc=[c for c in result['interventions'] if c['kind']==kind and c['frame']==frame and abs(c['dose'])==dose];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base])
    for mode in modes:
     vv=[c for c in cc if c['mode']==mode];result['summary'][f'{kind}_{frame}_{dose}_{mode}']=dict(internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),group_nll_abs_error=float(np.mean([abs(c['group_nll_error']) for c in vv])))
 replay=[abs(c['margin']-measure(captured_logits[c['id'],c['dose']],ex[c['id']]['groups'])['margin']) for c in result['interventions'] if c['mode']=='native']
 result['audits']=dict(old_span_containment_relative_error=containment,basis_refit=False,metric_training_cases=len(training),fresh_metric_training_cases=0,metric_projector_identity_error=float((K@A.T-torch.eye(64)).abs().max()),metric_system_condition=float(torch.linalg.cond(H)),metric_normal_equation_error=float((H@K.double()-cross).norm()/cross.norm()),oracle_normal_equation_max_relative_error=max(g['normal_equation_relative_error'] for g in oracle_geometry),oracle_coefficient_replay_max_relative_error=max(g['coefficient_replay_relative_error'] for g in oracle_geometry),oracle_worse_than_projection_count=sum(g['oracle_response_error']>g['projection_response_error']+1e-6 for g in oracle_geometry),context_switch_identity_max_relative_error=max(identityerrors),matched_input_norm_max_relative_error=max(normerrors,default=0.),reference_replay_max_margin_error=max(replay),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ('margin_error','reference_kl','internal_response_error')),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=bh==hashlib.sha256(basispath.read_bytes()).hexdigest(),number_basis_unchanged=cfg['number_basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(dict(vectors=vecs,coefficients=coefficients,examples=ex,oracle_geometry=oracle_geometry),ART/f'{STEM}_oracle.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()
