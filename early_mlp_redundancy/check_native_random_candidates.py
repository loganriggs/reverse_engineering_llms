"""M102 full training-document native/NumPy/override and finite-change preflight."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from inspect_model import load_model,trace
from native_random_candidate import NativeRandomCandidate
from independent_random_candidate import IndependentRandomCandidate
from native_mixed_quadratic import change

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'native_random_candidate_preflight_protocol.json';p=json.loads(pp.read_text())
 for name,h in p['source_hashes'].items():assert sha(V.parent/name)==h
 for name,h in p['references'].items():assert sha(V/name)==h
 entries=json.loads((V/'native_random_evaluation_models.json').read_text())['models'];cohort=json.loads((V/'natural_token_context_cohort.json').read_text());ids=torch.tensor([cohort['token_ids'][0]],device='cuda');model,receipt=load_model();mlp=model.transformer.h[1].mlp;base,cap=trace(model,ids);z=cap[1]['normalized_input'];nativewrite=mlp.Down(cap[1]['products']);rows=[];calls=1
 for entry in entries:
  enc=NativeRandomCandidate(entry);ind=IndependentRandomCandidate(entry);pred,support,code,info=enc.reconstruct(z,object());pe,ce=ind.verify(ind.prepare(z),pred,support,code)
  if entry['basis']=='native':assert torch.equal(pred,nativewrite)
  else:
   a=z[0,0].cpu().double().numpy();b=z[0,1].cpu().double().numpy();analytic=change(ind.L,ind.R,ind.D,ind.mean,ind.E,ind.B,a,b);actual=(pred[0,1].double()-pred[0,0].double()).cpu().numpy();err=float(np.linalg.norm(analytic-actual)/max(np.linalg.norm(analytic),1e-30));assert err<=p['finite_change_relative_tolerance']
  callback,_=trace(model,ids,layers=(0,2),replace=lambda j,x,w:enc(x)+mlp.Down_bias if j==1 else w);calls+=1
  original=mlp.forward;left,right,down=mlp.Left.forward,mlp.Right.forward,mlp.Down.forward
  def forbidden(*args,**kwargs):raise AssertionError('Native product modules must not be called by replacement')
  try:
   mlp.forward=lambda x:enc(x)+mlp.Down_bias;mlp.Left.forward=mlp.Right.forward=mlp.Down.forward=forbidden;override,_=trace(model,ids,layers=(0,2));calls+=1
  finally:mlp.forward=original;mlp.Left.forward=left;mlp.Right.forward=right;mlp.Down.forward=down
  difference=float((callback-override).abs().max());assert difference<=p['override_max_absolute_logit_difference']
  if entry['basis']=='native':assert torch.equal(base,callback)
  rows.append(dict(arm=entry['arm'],basis=entry['basis'],rank=entry['rank'],prediction_relative_error=pe,coordinate_relative_error=ce,override_max_logit_difference=difference,finite_change_relative_error=err if entry['basis']!='native' else None));print('CHECKED',entry['arm'],flush=True)
 restored,_=trace(model,ids,layers=(0,2));calls+=1;assert torch.equal(base,restored)
 out=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(pp),models_sha256=sha(V/'native_random_evaluation_models.json'),rows=rows,forward_count=calls,checkpoint_sha256=receipt.weights_sha256,native_control_exact=True,native_restoration_exact=True,scope='All11trainingdoc0 predictions independentlycheckedwithNumPy; finitechangesatfirsttwopositions, nativecontrolbitexact, fullMLPoverrideguardedLeft/Right/Downmodules. Privatecopiesofnativeweightsstillused; no compression orspeedclaim.')
 f=V/'native_random_candidate_preflight.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls)
if __name__=='__main__':main()
