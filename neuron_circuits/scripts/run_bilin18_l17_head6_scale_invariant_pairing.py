"""H355 scale-free within-site pairing and unnormalized pooled comparison."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM='bilin18_l17_head6_scale_invariant_pairing'
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM}.json').read_text());assert hashlib.sha256((ART/cfg['source']).read_bytes()).hexdigest()==cfg['sha256'];z=np.load(ART/cfg['source']);ray=z['rayleigh'].reshape(7,13,2,128,4);scale=z['reference_trace'][z['indices']].reshape(7,1,1,128,4);raw=ray*scale;assert ray.min()>0
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));ranks=rng.integers(0,6,(4000,6));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']};summary={};boots={};audit=0.
 def statistic(v,start,kind):
  A=v[:,0];C=v[:,start:start+6].mean(1)
  return np.exp(np.log(C/A).mean()) if kind=='invariant' else C.mean()/A.mean()
 for path,label in enumerate(['compiled','native_mlp']):
  summary[label]={}
  for kind,data in [('invariant',ray),('raw_pooled',raw)]:
   summary[label][kind]={}
   for n,start in [('moment',1),('donor',7)]:
    v=data[:,:,path];own=statistic(v[:1],start,kind);matched=statistic(v[1:],start,kind);values=np.array([own,matched,own/matched]);b=[]
    if kind=='invariant':audit=max(audit,abs(own-statistic(raw[:1,:,path],start,kind)),abs(matched-statistic(raw[1:,:,path],start,kind)))
    for ix,rr,mr in zip(docs,draws[n],ranks):
     A=v[:,0][:,ix];C=v[:,start+rr][:,:,ix].mean(1)
     if kind=='invariant':o=np.exp(np.log(C[:1]/A[:1]).mean());m=np.exp(np.log(C[1+mr]/A[1+mr]).mean())
     else:o=C[:1].mean()/A[:1].mean();m=C[1+mr].mean()/A[1+mr].mean()
     b.append([o,m,o/m])
    b=np.array(b);ci=np.quantile(b,[.025,.975],axis=0);summary[label][kind][n]={k:{'estimate':float(values[i]),'ci95':ci[:,i].tolist()} for i,k in enumerate(['own','matched','own_over_matched'])};boots[label+'_'+kind+'_'+n]=b
 assert audit<1e-10
 gates={f'native_{kind}_pairing':all(summary['native_mlp'][key][n]['own_over_matched']['ci95'][0]>1.2 for n in ['moment','donor']) for kind,key in [('invariant','invariant'),('raw','raw_pooled')]}
 np.savez_compressed(ART/f'{STEM}_bootstrap.npz',**boots);(ART/f'{STEM}.json').write_text(json.dumps({'config':cfg,'summary':summary,'gates':gates,'normalization_invariance_error':audit},indent=2)+'\n');print(summary,gates,'audit',audit)
if __name__=='__main__':main()
