"""M79 executable whole-MLP replacement and factored-polynomial preflight."""
import json
import torch
import numpy as np
from inspect_model import load_model,trace
from local_affine_chart_predictor import LocalAffineChartPredictor
from fit_pca_encoder import sha,V
P=V.parent

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 cohort=json.loads((V/'natural_token_context_cohort.json').read_text());ids=torch.tensor([cohort['token_ids'][0]],device='cuda');model,receipt=load_model();mlp=model.transformer.h[1].mlp;native,cap=trace(model,ids);z=cap[1]['normalized_input'];x=z[0].cpu().double().numpy();L,R,D=[m.weight.cpu().double().numpy() for m in [mlp.Left,mlp.Right,mlp.Down]];rows=[]
 for name in ['mean','random32','kmeans32']:
  enc=LocalAffineChartPredictor(name);pred=enc(z);labels=np.sum((x[:,None,:]-enc.anchors[None,:,:])**2,axis=2).argmin(1);center=enc.anchors[labels];dx=x-center
  exact=((center@L.T)*(center@R.T)+(center@L.T)*(dx@R.T)+(dx@L.T)*(center@R.T))@D.T
  error=np.linalg.norm(pred[0].cpu().double().numpy()-exact)/max(1,np.linalg.norm(exact));assert error<1e-5
  callback,_=trace(model,ids,layers=(0,2),replace=lambda j,z,w:enc(z)+mlp.Down_bias if j==1 else w)
  old,oldleft=mlp.forward,mlp.Left.forward
  def forbidden(*a,**kw):raise AssertionError('Native product branch called')
  try:
   mlp.forward=lambda z:enc(z)+mlp.Down_bias;mlp.Left.forward=forbidden;out,_=trace(model,ids,layers=(0,2))
  finally:mlp.forward=old;mlp.Left.forward=oldleft
  assert torch.equal(out,callback);rows.append(dict(name=name,applied_prediction_relative=error,callback_override_max_difference=float((out-callback).abs().max()),product_guard_passed=True))
 restored,_=trace(model,ids,layers=(0,2));assert torch.equal(native,restored)
 result=dict(status='passed',source_sha256=sha(__file__),predictor_source_sha256=sha(P/'local_affine_chart_predictor.py'),fit_audit_sha256=sha(V/'local_affine_chart_fit_audit.json'),checkpoint_sha256=receipt.weights_sha256,forward_count=8,training_document=0,native_restored_exact=True,models=rows)
 dest=V/'local_affine_chart_forward_preflight.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
