"""Training-only M92 algebra, access guards, native replay and intervention preflight."""
import json
from pathlib import Path
import numpy as np
import torch
from fit_pca_encoder import sha,V,A
from inspect_model import load_model,trace
from product_subset_predictor import ProductSubsetPredictor,intervention_delta
P=Path(__file__).resolve().parent
@torch.no_grad()
def run():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
 manifest=json.loads((V/'product_subset_models.json').read_text());assert manifest['encoder_source_sha256']==sha(P/'product_subset_predictor.py');assert manifest['protocol_sha256']==sha(V/'product_subset_causal_protocol.json')
 model,receipt=load_model();mlp=model.transformer.h[1].mlp;calls=0
 cohort=json.loads((V/'natural_token_context_cohort.json').read_text());ids=torch.tensor([cohort['token_ids'][0]],device='cuda');base,cap=trace(model,ids);calls+=1;z=cap[1]['normalized_input'];nativewrite=mlp.Down(cap[1]['products'])
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True);L=r['L'].double().numpy();R=r['R'].double().numpy();W=r['D'].double().numpy();zn=z.cpu().double().numpy();H=(zn@L.T)*(zn@R.T)
 pair=json.loads((V/'training_causal_coverage_protocol.json').read_text())['pairs'][0];pairids=[torch.tensor([t],device='cuda') for t in pair['token_ids']];pairz=[];pairw=[];pairbase=[]
 for pi in pairids:
  out,c=trace(model,pi);calls+=1;pairz.append(c[1]['normalized_input']);pairw.append(mlp.Down(c[1]['products']));pairbase.append(out[0,-1])
 records=[x for x in json.loads((V/'training_causal_coverage.json').read_text())['records'] if x['pair_id']==pair['id'] and x['layer']==1 and x['status']=='evaluated'];assert len(records)==6
 def forbidden(*args,**kwargs):raise AssertionError('Predictor accessed native MLP module')
 rows=[];constructions=0;smoke=[];nativeKL=None;guardcount=0
 for e in manifest['models']:
  enc=ProductSubsetPredictor(e);handles=[m.register_forward_pre_hook(forbidden) for m in [mlp.Left,mlp.Right,mlp.Down]]
  try:
   pred,support,code,info=enc.reconstruct(z,object());assert torch.equal(support,torch.ones_like(support));guardcount+=1
  finally:
   for h in handles:h.remove()
  ix=np.array(e['indices']);direct=(H[...,ix]*e['scalar'])@W[:,ix].T
  if e['space']=='native':expected=direct;fullref=nativewrite.cpu().double().numpy()
  else:
   assert sha(e['state_path'])==e['state_sha256'];s=torch.load(e['state_path'],weights_only=True)['state'];D=s['features.dictionary'].double().numpy();U,sv,_=np.linalg.svd(D,full_matrices=False);assert min(sv)>max(sv)*1e-10;c=s['output_mean'].double().numpy()+float(s['output_scale'])*s['features.bias'].double().numpy();offset=c-(c@U)@U.T;expected=(direct@U)@U.T+offset;fullref=c+((nativewrite.cpu().double().numpy()-c)@U)@U.T
  actual=pred.cpu().double().numpy();relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected));assert relative<=1e-5
  cr=float(np.linalg.norm(code.cpu().double().numpy()-H[...,ix])/np.linalg.norm(H[...,ix]));assert cr<=1e-5
  row={k:e[k] for k in ['space','method','correction','size','scalar']};row.update(relative_write_error=relative,relative_product_error=cr,max_absolute_write_error=float(abs(actual-expected).max()))
  if e['method']=='full':
   frel=float(np.linalg.norm(actual-fullref)/np.linalg.norm(fullref));row['full_reference_relative_error']=frel;assert frel<=1e-5
  if e['space']=='native' and e['method']=='full':
   # Replace native forward itself; trace cannot compute a native layer1 write first.
   original=mlp.forward;mlp.forward=lambda zz:enc.reconstruct(zz)[0]+mlp.Down_bias
   handles=[m.register_forward_pre_hook(forbidden) for m in [mlp.Left,mlp.Right,mlp.Down]]
   try:out,_=trace(model,ids,layers=(0,2));calls+=1
   finally:
    mlp.forward=original
    for h in handles:h.remove()
   lp=base.double().log_softmax(-1);lq=out.double().log_softmax(-1);nativeKL=float((lp.exp()*(lp-lq)).sum(-1).mean());assert abs(nativeKL)<=1e-8;row['native_full_KL']=nativeKL
  pq=[enc.reconstruct(zz)[0] for zz in pairz]
  runsmoke=(e['space']=='native' and e['method'] in ['full','energy']) or (e['space']=='dense_71_intact_KL' and e['method']=='full')
  smoke_bases=[]
  if runsmoke:
   for ep in range(2):
    out,_=trace(model,pairids[ep],replace=lambda j,zz,w:pq[ep]+mlp.Down_bias if j==1 else w);calls+=1;smoke_bases.append(out[0,-1])
  for rec in records:
   ep=rec['endpoint'];de=1-ep;pos=rec['recipient_positions'];dp=rec['donor_positions'];region=rec['region']
   delta=intervention_delta(pq[ep],pq[de],pos,dp);nd=intervention_delta(pairw[ep],pairw[de],pos,dp)
   expected_delta=np.zeros(tuple(pq[ep].shape),dtype=np.float32)
   for i,j in zip(pos,dp):expected_delta[:,i]=pq[de][:,j].cpu().numpy()-pq[ep][:,i].cpu().numpy()
   assert np.array_equal(delta.cpu().numpy(),expected_delta)
   if region=='identity':assert not bool(delta.count_nonzero()) and not bool(nd.count_nonzero())
   for corner,bg,change in [('native_reconstructed',pairw[ep],delta),('reconstructed_native',pq[ep],nd),('reconstructed_reconstructed',pq[ep],delta)]:
    target=bg+change;outside=sorted(set(range(bg.shape[1]))-set(pos));assert torch.equal(target[:,outside],bg[:,outside]);constructions+=1
    if runsmoke:
     baseline=pairbase[ep] if corner=='native_reconstructed' else smoke_bases[ep]
     out,_=trace(model,pairids[ep],replace=lambda j,zz,w:target+mlp.Down_bias if j==1 else w);calls+=1;changed=out[0,-1]
     assert torch.isfinite(changed).all()
     if region=='identity':assert torch.equal(changed,baseline)
     smoke.append({'space':e['space'],'method':e['method'],'correction':e['correction'],'endpoint':ep,'region':region,'corner':corner,'max_logit_change':float(abs(changed-baseline).max()),'identity_exact':bool(torch.equal(changed,baseline)) if region=='identity' else None})
  rows.append(row);del enc
  print('checked',e['space'],e['method'],e['correction'],relative,flush=True)
 out={'status':'passed','source_sha256':sha(__file__),'models_sha256':sha(V/'product_subset_models.json'),'encoder_sha256':sha(P/'product_subset_predictor.py'),'protocol_sha256':sha(V/'product_subset_causal_protocol.json'),'native_checkpoint_sha256':receipt.weights_sha256,'training_document':0,'training_pair':pair['id'],'prediction_checks':rows,'input_access_guards_passed':guardcount,'intervention_constructions_checked':constructions,'native_smoke_conditions':smoke,'native_full_KL':nativeKL,'forward_count':calls,'scope':'All119 input-only predictors and 17full references checked on training doc0. All119x2endpointsx3regionsx3corners construction checks. Native smoke uses fixed native full/energy raw/scaled and dense71KLfull controls; not a substitute for heldout evaluation or full response-vector audit.'}
 assert len(rows)==119 and constructions==2142 and guardcount==119
 (V/'product_subset_preflight.json').write_text(json.dumps(out,indent=2)+'\n');print('PASSED',calls,flush=True)
if __name__=='__main__':run()
