"""Frozen M99 scalar and coefficient-gradient linearity checks, including missing source."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,A,sha
from bilinear_input_causal_loss import InputCausalExperiment
from product_dictionary_featurizer import ProductDictionaryFeaturizer
from product_dictionary_component_gradients import components,gradient

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'product_dictionary_component_preflight_protocol.json';p=json.loads(pp.read_text())
 for n,h in p['source_hashes'].items():assert sha(V.parent/n)==h
 for n,h in p['references'].items():assert sha(V/n)==h
 e=InputCausalExperiment();rows=[];samples=[]
 for entry in p['models']:
  assert sha(entry['path'])==entry['sha256'];f=ProductDictionaryFeaturizer(torch.load(entry['path'],weights_only=True)['state']).cuda()
  for key in p['keys']:
   K,B,parts,count=components(e,f,key);gK=gradient(K,f);gs={n:gradient(x,f) for n,x in parts.items()};gB=gradient(B,f);gTotal=gradient(K+p['joint_weight']*B,f)
   scalars={n:float(x.detach()) if x is not None else 0. for n,x in parts.items()};scalar_error=abs(sum(scalars.values())-float(B.detach()));assert scalar_error<=p['scalar_atol']+p['scalar_rtol']*abs(float(B.detach()))
   pairs=[('B',gB,sum(gs.values())),('K_plus_lambda_B',gTotal,gK+p['joint_weight']*sum(gs.values()))];checks=[]
   for name,direct,assembled in pairs:
    dn=direct.numpy();sn=assembled.numpy();error=float(np.linalg.norm(dn-sn));norm=float(np.linalg.norm(dn));allowed=p['gradient_atol']+p['gradient_rtol']*norm;assert error<=allowed,(entry['seed'],key,name,error,allowed);checks.append(dict(name=name,norm=norm,absolute_error=error,relative_error=error/max(norm,1e-300),allowed=allowed))
   assert all(x.grad is None for x in e.model.parameters())
   rows.append(dict(seed=entry['seed'],key=key,nonidentity_branches=count,K=float(K.detach()),B=float(B.detach()),parts=scalars,scalar_error=scalar_error,gradient_checks=checks));samples.append(dict(row=len(rows)-1,K=gK,B=gB,total=gTotal,parts=gs));print('CHECKED',entry['seed'],key,count,flush=True)
   del K,B,parts,gK,gs,gB,gTotal,pairs,f
   f=ProductDictionaryFeaturizer(torch.load(entry['path'],weights_only=True)['state']).cuda()
 path=A/'emlp_m99_component_preflight.pt';assert not path.exists();torch.save(samples,path)
 out=dict(status='passed',protocol_sha256=sha(pp),source_sha256=sha(__file__),rows=rows,forward_count=e.calls,samples_path=str(path),samples_sha256=sha(path),scope='Scalar arithmetic independently summed; coefficient gradient linearity checked in NumPy against separately differentiated aggregate. Same autograd/nativeTransformer, not an independent derivative implementation. Float32VJP tolerance frozen in advance. Missing-source divisor preserved.')
 dest=V/'product_dictionary_component_preflight.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('DONE',len(rows),e.calls)
if __name__=='__main__':main()
