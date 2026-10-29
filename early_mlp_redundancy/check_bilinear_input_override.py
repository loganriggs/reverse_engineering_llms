"""Untrained executable featurizers bypass native MLP products exactly."""
import json
import torch
from bilinear_input_runtime import InputWritePredictor
from inspect_model import load_model,trace
from fit_pca_encoder import sha,V
P=V.parent

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);manifest=json.loads((V/'bilinear_input_initial_states.json').read_text());assert manifest['module_sha256']==sha(P/'bilinear_input_featurizer.py');model,receipt=load_model();mlp=model.transformer.h[1].mlp;ids=torch.tensor([json.loads((V/'natural_token_context_cohort.json').read_text())['token_ids'][0]],device='cuda');native,_=trace(model,ids,layers=(0,2));rows=[]
 for entry in manifest['models']:
  assert sha(entry['path'])==entry['sha256'];predict=InputWritePredictor(torch.load(entry['path'],weights_only=True)).cuda();callback,_=trace(model,ids,layers=(0,2),replace=lambda j,z,w:predict(z)[0]+mlp.Down_bias if j==1 else w)
  original,originalleft,originalright=mlp.forward,mlp.Left.forward,mlp.Right.forward
  def forbidden(*a,**kw):raise AssertionError('Native product branch was called')
  try:
   mlp.forward=lambda z:predict(z)[0]+mlp.Down_bias;mlp.Left.forward=forbidden;mlp.Right.forward=forbidden;out,_=trace(model,ids,layers=(0,2))
  finally:mlp.forward=original;mlp.Left.forward=originalleft;mlp.Right.forward=originalright
  assert torch.isfinite(out).all() and torch.equal(out,callback);rows.append(dict(kind=entry['kind'],seed=entry['seed'],max_logit_difference=float((out-callback).abs().max()),product_guards_passed=True))
 restored,_=trace(model,ids,layers=(0,2));assert torch.equal(native,restored)
 result=dict(status='passed',source_sha256=sha(__file__),runtime_sha256=sha(P/'bilinear_input_runtime.py'),initial_states_sha256=sha(V/'bilinear_input_initial_states.json'),checkpoint_sha256=receipt.weights_sha256,forward_count=18,training_document=0,native_restored_exact=True,models=rows,scope='All eight untrained predictors execute using only normalized inputs and frozen statistics; both native product branches guarded; callback/override exact. Not evidence of predictive quality or downstream gradient correctness.')
 out=V/'bilinear_input_forward_override.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(rows),'overrides',18,'forwards')
if __name__=='__main__':main()
