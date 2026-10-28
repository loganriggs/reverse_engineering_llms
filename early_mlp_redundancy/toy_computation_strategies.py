"""Known-ground-truth bilinear toy: identical clustered function, two channel implementations.
This validates diagnostic logic, not an explanation of a trained language model.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 rng=np.random.default_rng(2026102902)
 # x=[one-hot token, constant1]; h_k = L_k(x)*R_k(x), R_k(x)=1.
 tokens=64;classes=4;copies=8;width=classes*copies;dim=8
 x=np.concatenate([np.eye(tokens),np.ones((tokens,1))],axis=1)
 centers=np.linalg.qr(rng.normal(size=(dim,classes)))[0].T
 assignment=np.arange(tokens)%classes
 L=np.zeros((width,tokens+1));R=np.zeros_like(L);R[:,-1]=1
 for k in range(classes):L[k*copies:(k+1)*copies,:tokens]=(assignment==k)[None,:]
 h=(x@L.T)*(x@R.T)
 decoders={}
 decoders['duplicated']=np.repeat(centers,copies,axis=0)/copies
 signs=np.tile(np.array([1,-1]*4),classes)
 decoders['cancelling']=decoders['duplicated']*(1+4*signs[:,None])
 target=centers[assignment];results={};outputs={}
 masks={}
 for fraction in (.25,.5,.875):
  nremove=int(width*fraction);masks[fraction]=[]
  for seed in range(128):
   chosen=np.random.default_rng(1000+seed).permutation(width)[:nremove]
   keep=np.ones(width);keep[chosen]=0;masks[fraction].append(keep)
 for name,D in decoders.items():
  writes=h[:,:,None]*D[None,:,:];y=writes.sum(1);outputs[name]=y
  assert np.max(np.abs(y-target))<1e-12
  cancellation=float((np.linalg.norm(writes,axis=-1).sum(1)/np.linalg.norm(y,axis=-1)).mean())
  curves=[]
  for fraction,mm in masks.items():
   errors=[];cosines=[];norms=[]
   for keep in mm:
    edited=(writes*keep[None,:,None]).sum(1);errors.append(float(np.square(edited-y).sum()/np.square(y).sum()))
    norms.append(float((np.linalg.norm(edited,axis=-1)/np.linalg.norm(y,axis=-1)).mean()))
    den=np.linalg.norm(edited,axis=-1)*np.linalg.norm(y,axis=-1)
    cosines.append(float(np.divide((edited*y).sum(-1),den,out=np.zeros_like(den),where=den>0).mean()))
   curves.append(dict(removed_fraction=fraction,mean_relative_squared_write_error=float(np.mean(errors)),min_error=min(errors),max_error=max(errors),mean_norm_ratio=float(np.mean(norms)),mean_cosine_zero_if_zero=float(np.mean(cosines))))
  results[name]=dict(cancellation_ratio=cancellation,curves=curves)
 assert np.max(np.abs(outputs['duplicated']-outputs['cancelling']))<1e-12
 assert abs(results['duplicated']['cancellation_ratio']-1)<1e-12
 assert abs(results['cancelling']['cancellation_ratio']-4)<1e-12
 # Arbitrary within-class averaging preserves both implementations' output exactly.
 for y in outputs.values():
  clustered=np.stack([y[assignment==k].mean(0) for k in range(classes)])[assignment]
  assert np.max(np.abs(y-clustered))<1e-12
 out=dict(status='known_ground_truth_checks_passed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=2026102902,tokens=tokens,classes=classes,width=width,output_dimension=dim,same_function_max_error=float(np.max(np.abs(outputs['duplicated']-outputs['cancelling']))),results=results,
 interpretation='Exact clustered token map implemented either by positive duplicate channels or large opposing channels. Same widths, outputs and masks. Clustering and cancellation are not mutually exclusive layer types. Toy local errors are not LLM output KL; no trained-model match is claimed.')
 args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
