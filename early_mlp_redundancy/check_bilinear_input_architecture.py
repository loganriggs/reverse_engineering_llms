"""Independent polynomial, shared-union and gradient checks before any fit."""
import json
import numpy as np
import torch
from bilinear_input_featurizer import BilinearInputFeaturizer,KINDS
from fit_pca_encoder import sha,V
P=V.parent

def main():
 torch.set_num_threads(1);pp=V/'bilinear_input_architecture_check_protocol.json';protocol=json.loads(pp.read_text());tol=protocol['tolerances'];assert protocol['protocol_sha256']==sha(V/'bilinear_input_featurizer_preflight_protocol.json')
 g=torch.Generator().manual_seed(8101);x=torch.randn(8,1152,generator=g,dtype=torch.float64);target=torch.randn(8,1152,generator=g,dtype=torch.float64);results=[]
 for seed in [71,72]:
  baseline=BilinearInputFeaturizer('dense',seed)
  for kind in KINDS:
   model=BilinearInputFeaturizer(kind,seed)
   assert sum(p.numel() for p in model.parameters())==443776
   for (n,p),(n2,q) in zip(model.named_parameters(),baseline.named_parameters()):assert n==n2 and torch.equal(p,q)
   model=model.double();p={k:v.detach().numpy() for k,v in model.named_parameters()};M=model.membership.numpy();xx=x.numpy();a=(xx@p['left'].T+p['left_bias'])*(xx@p['right'].T+p['right_bias'])
   def independent(a):
    supports=[]
    for row in a:
     if kind=='dense':chosen=set(range(128))
     elif kind=='scalar':chosen=set(sorted(range(128),key=lambda i:(-abs(row[i]),i))[:32])
     else:
      scores=[sum(row[j]**2 for j in range(128) if M[i,j]) for i in range(32)];groups=sorted(range(32),key=lambda i:(-scores[i],i))[:8 if kind=='block' else 4];chosen={j for i in groups for j in range(128) if M[i,j]}
     supports.append([float(j in chosen) for j in range(128)])
    return np.array(supports)
   support=independent(a);D=p['dictionary']/np.sqrt((p['dictionary']**2).sum(0));ref=(a*support)@D.T+p['bias'];out,q,code=model(x);assert np.array_equal(support,q.numpy());error=np.linalg.norm(out.detach().numpy()-ref)/max(1,np.linalg.norm(ref));assert error<tol['output_relative']
   if kind=='shared_fixed':assert np.all(M.sum(0)==2) and np.all(M.sum(1)==8)
   zeroq,_=model.select(torch.zeros_like(code));assert np.array_equal(zeroq.numpy(),independent(np.zeros_like(a)))
   loss=(out-target).square().mean();params=list(model.parameters());grads=torch.autograd.grad(loss,params);original=[p.detach().clone() for p in params];checks=[]
   for direction in range(4):
    vs=[torch.randn(p.shape,generator=g,dtype=p.dtype) for p in params];norm=torch.sqrt(sum(v.square().sum() for v in vs));vs=[v/norm for v in vs];ad=float(sum((v*grad).sum() for v,grad in zip(vs,grads)));vals=[]
    for sign in [-1,1]:
     with torch.no_grad():
      for p,start,v in zip(params,original,vs):p.copy_(start+sign*tol['finite_difference_epsilon']*v)
      y,qq,_=model(x);assert torch.equal(qq,q),'Support crossing in fixed gradient case';vals.append(float((y-target).square().mean()))
    with torch.no_grad():
     for p,start in zip(params,original):p.copy_(start)
    fd=(vals[1]-vals[0])/(2*tol['finite_difference_epsilon']);assert np.isclose(fd,ad,rtol=tol['gradient_relative'],atol=tol['gradient_absolute']),(kind,ad,fd);checks.append(dict(autograd=ad,finite_difference=fd,absolute_error=abs(ad-fd)))
   # Exactly represented rank-one bilinear teacher; shared groups contain atom0 twice.
   with torch.no_grad():
    for p in model.parameters():p.zero_()
    model.dictionary.copy_(torch.eye(1152,128,dtype=torch.float64));model.left[0,0]=1;model.left_bias[0]=.5;model.right[0,1]=2;model.right_bias[0]=-.3
   prediction,qq,cc=model(x);expected=torch.zeros_like(prediction);expected[:,0]=(x[:,0]+.5)*(2*x[:,1]-.3);teachererr=float((prediction-expected).abs().max());assert teachererr<tol['teacher_absolute']
   raw=torch.zeros(8,128,dtype=torch.float64);raw[:,0]=expected[:,0];_,gates=model.select(raw)
   if kind=='shared_fixed':assert bool(((gates@model.membership)[:,0]==2).all()) and bool((qq[:,0]==1).all())
   results.append(dict(kind=kind,seed=seed,parameters=443776,independent_output_relative=error,teacher_max_absolute=teachererr,active_counts=support.sum(1).tolist(),gradient_checks=checks))
 result=dict(status='passed',source_sha256=sha(__file__),module_sha256=sha(P/'bilinear_input_featurizer.py'),protocol_sha256=sha(pp),models=results,scope='Independent numerical architecture checks and exact injected-teacher recovery; not optimizer recovery, semantic identifiability or native-model loss validation. No optimizer updates.')
 dest=V/'bilinear_input_architecture_checks.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(results),'models',sum(len(r['gradient_checks']) for r in results),'gradient directions')
if __name__=='__main__':main()
