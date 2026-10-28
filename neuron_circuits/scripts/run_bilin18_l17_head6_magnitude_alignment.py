"""H343 exact aggregate magnitude/alignment accounting; no causal swaps."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM='bilin18_l17_head6_magnitude_alignment'
def decompose(A,C):
 def parts(G):
  a,b=G[0,0],G[1,1];return a+b,2*np.sqrt(a*b),G[0,1]/np.sqrt(a*b)
 sa,pa,ra=parts(A);sc,pc,rc=parts(C)
 mag=sc-sa+(pc-pa)*(ra+rc)/2
 align=(rc-ra)*(pa+pc)/2
 delta=C[:2,:2].sum()-A[:2,:2].sum()
 assert np.isclose(mag+align,delta,rtol=1e-10,atol=1e-10)
 return np.array([delta,mag,align,mag/delta,align/delta,ra,rc])
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM}.json').read_text());assert hashlib.sha256((ART/cfg['source']).read_bytes()).hexdigest()==cfg['sha256'];z=np.load(ART/cfg['source']);rng=np.random.default_rng(cfg['bootstrap_seed']);summary={};draws={}
 # Same document and draw indices for all scopes/contrasts; donor pairing is conditioned on fixed assignments.
 docs=rng.integers(0,128,(cfg['bootstrap_samples'],128));rep={n:rng.integers(0,6,(cfg['bootstrap_samples'],6)) for n in ['moment','donor']}
 for scope in ['all','top4']:
  A=z[f'actual_{scope}_term_gram'];summary[scope]={}
  for name in ['moment','donor']:
   C=np.stack([z[f'{name}{i}_{scope}_term_gram'] for i in range(6)]);v=decompose(A.sum(0),C.mean(0).sum(0));boot=np.stack([decompose(A[ix].sum(0),C[rr][:,ix].mean(0).sum(0)) for ix,rr in zip(docs,rep[name])]);ci=np.quantile(boot,[.025,.975],axis=0);draws[f'{scope}_{name}']=boot
   keys=['tangent_excess','magnitude_contribution','alignment_contribution','magnitude_fraction','alignment_fraction','actual_cosine','control_cosine'];summary[scope][name]={k:{'estimate':float(v[i]),'ci95':ci[:,i].tolist()} for i,k in enumerate(keys)}
 gates={k+'_majority':all(summary['all'][n][k+'_fraction']['ci95'][0]>.5 for n in ['moment','donor']) for k in ['alignment','magnitude']}
 (ART/f'{STEM}.json').write_text(json.dumps({'config':cfg,'summary':summary,'gates':gates},indent=2)+'\n');np.savez_compressed(ART/f'{STEM}_bootstrap.npz',**draws);print(json.dumps({'summary':summary,'gates':gates},indent=2))
if __name__=='__main__':main()
