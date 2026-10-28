"""H716: anchor-fixed exact-Gram controls in the measured decoder span."""
import json,time
from pathlib import Path
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_exact_geometry_controls'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'previous':ART/'bilin18_group_response_geometry.json','data':ART/'bilin18_coactivation_replication_captures.pt','groups':ART/'bilin18_group_neighbor_controls_frozen_controls.json','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','rotations':ART/'bilin18_early_sparse_rotations_captures.pt','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'}
 paths.update({f'response_{i}':ART/'group_response_geometry'/f'context_{i:02d}.pt' for i in range(16)});hashes={k:sha(p) for k,p in paths.items()};old=json.loads(paths['previous'].read_text());groups=json.loads(paths['groups'].read_text());dirs=old['config']['directions'];ix={v:i for i,v in enumerate(dirs)};sp=torch.load(paths['spectra'],weights_only=False)['states']['residual1'];Q=torch.load(paths['rotations'],weights_only=False)['rotations']['residual1','learned',7061].double();D=(sp['eigenvectors'][:,:256].double()*sp['eigenvalues'][:256].double().sqrt())@Q;D/=D.norm(dim=0);D=D[:,dirs];U,T=torch.linalg.qr(D);Ti=torch.linalg.inv(T);gen=torch.Generator().manual_seed(716);controls=[];geomerrors=[]
 for g in groups:
  members=[ix[x] for x in g['members']];a=T[:,members[0]];a=a/a.norm();N=T[:,members[1:]];alpha=a@N;perp=N-a[:,None]*alpha;W,R=torch.linalg.qr(perp);assert torch.linalg.matrix_rank(perp)==7;frames=[]
  for repeat in range(199):
   Z=torch.randn(len(dirs),7,generator=gen,dtype=torch.double);Z-=a[:,None]*(a@Z);Z,rz=torch.linalg.qr(Z);Z*=torch.diag(rz).sign();rotated=a[:,None]*alpha+Z@R;C=Ti@rotated;frames.append(C);geomerrors.append(float(((D@C).T@(D@C)-D[:,members[1:]].T@D[:,members[1:]]).abs().max()));geomerrors.append(float(((D[:,members[0]]@(D@C))-(D[:,members[0]]@D[:,members[1:]])).abs().max()))
  controls.append(torch.stack(frames))
 cfg=dict(experiment='H716',hashes=hashes,design='199joint controlpanels across8frozengroups inH715measured134Ddecoder span. For eachanchor holdits directionfixed; preserve all7neighbor pairwiseinnerproducts andanchor-neighbor innerproducts by replacing orthogonal-residual7framewithHaar-random7frameinanchorperpendicularspan. Seed716. Same rotations acrossall16contexts. Reuse symmetricfinite-difference derivatives tocompute Fisher andconfidence-removed squaredresponsecosines. Compareactualmean to199panelmeans andpairedcontextbootstrap. Directly verify firstcontrol firstneighbor for each8groups in first2contexts at+/-1%residualnorm. No outcome selection.',limitations='Rotations preserve exactgeometry but notnaturalactivationfrequency/support orsemanticdata support. Restricted134Dmeasuredspan; local linear response. Reuses16H715contexts; exploratorymodel-conditionednullnotuniversalhypothesistest. Originalreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));torch.save(dict(controls=controls,directions=dirs,decoder=D),ART/f'{STEM}_controls.pt');records=[];predicted=[]
 for ci in range(16):
  state=torch.load(paths[f'response_{ci}'],weights_only=False);response=state['response'].double();l=state['native_logits'].double();p=l.softmax(-1);sig=(response-(response*p).sum(-1,keepdim=True))*p.sqrt();temp=(l-(l*p).sum())*p.sqrt();temp/=temp.norm();clean=sig-(sig@temp)[:,None]*temp
  for mode,X in [('fisher',sig),('confidence_removed',clean)]:
   F=X@X.T
   for gi,g in enumerate(groups):
    ai=ix[g['anchor']];nei=[ix[x] for x in g['members'][1:]];C=controls[gi];dots=torch.einsum('i,rik->rk',F[ai],C);norms=torch.einsum('rik,ij,rjk->rk',C,F,C);score=(dots.square()/(F[ai,ai]*norms).clamp_min(1e-30)).mean(-1);actual=float((F[ai,nei].square()/(F[ai,ai]*F.diag()[nei])).mean());records.append(dict(context=ci,doc=state['doc'],anchor=g['anchor'],mode=mode,actual=actual,control_scores=score.tolist(),difference=actual-float(score.mean())))
  if ci<2:
   for gi,g in enumerate(groups):predicted.append(dict(context=ci,doc=state['doc'],anchor=g['anchor'],coeff=controls[gi][0,:,0],response=controls[gi][0,:,0]@response,epsilon=state['epsilon'],native_logits=l))
  print('GEOMETRY',ci+1,flush=True)
 data=torch.load(paths['data'],weights_only=False);model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 checks=[];forwards=0
 for ci in range(2):
  subset=[r for r in predicted if r['context']==ci];patch=torch.stack([(D@r['coeff'])*r['epsilon'] for r in subset]).float().cuda();patch=torch.cat([patch,-patch]);ids=data['all_token_ids'][subset[0]['doc']:subset[0]['doc']+1,:64].cuda().expand(16,-1)
  def edit(m,args,out):
   x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
  h=model.blocks[1].register_forward_hook(edit)
  try:logits,_=run(model,ids)
  finally:h.remove()
  forwards+=16
  for k,r in enumerate(subset):
   actual=(logits[k].double().cpu()-logits[k+8].double().cpu())/(2*r['epsilon']);p=r['native_logits'].softmax(-1);pred=r['response'];a=(actual-(actual*p).sum())*p.sqrt();b=(pred-(pred*p).sum())*p.sqrt();checks.append(dict(context=ci,anchor=r['anchor'],signature_relative_error=float((a-b).norm()/a.norm()),signature_cosine=float((a@b)/(a.norm()*b.norm()))))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},exact_decoder_gram_max_abs_error=max(geomerrors),decoder_condition_number=float(torch.linalg.cond(D)),finite=all(np.isfinite(r['difference']) for r in records),direct_checks=checks,whole_model_forwards=forwards,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and max(geomerrors)<1e-10 and audits['finite'];(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
