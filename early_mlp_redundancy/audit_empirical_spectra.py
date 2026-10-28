"""Independent NumPy spectral and source-prefix audit for M2."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation';path=P/'empirical_spectra.json';r=json.loads(path.read_text());panel=json.loads((P/'training_source_context_audit.json').read_text());saved=torch.load('/tmp/emlp_m2_matrices.pt',weights_only=True)
assert r['forward_count']==160 and len(r['results'])==12
expected=[]
for pair in panel['rows']:
 for ep in (0,1):
  for pos in range(len(pair['token_ids'][ep])):expected.append((pair['id'],ep,pos,pair['token_ids'][ep][:pos+1]))
assert len(expected)==r['positions']==len(r['occurrences'])
for (ident,ep,pos,prefix),o in zip(expected,r['occurrences']):
 assert (ident,ep,pos)==(o['id'],o['endpoint'],o['position']) and prefix==r['prefixes'][o['prefix_id']]
assert len({tuple(x) for x in r['prefixes']})==r['unique_prefix_count']==len(r['unique_indices'])
for i,j in enumerate(r['unique_indices']):assert r['occurrences'][j]['prefix_id']==i and j==next(k for k,o in enumerate(r['occurrences']) if o['prefix_id']==i)
errors=[];spectral=[]
for s in saved:
 H=s['H'].double().numpy();Y=s['Y'].double().numpy();cn=s['column_squared_norms'].numpy()
 for sampling in ('occurrences','unique_prefixes'):
  ind=np.arange(len(Y)) if sampling=='occurrences' else np.array(r['unique_indices']);h=H[ind];y=Y[ind]
  for centered in (False,True):
   hh=h-h.mean(0) if centered else h;yy=y-y.mean(0) if centered else y;sv=np.linalg.svd(yy,compute_uv=False);v=next(v for v in r['results'] if (v['layer'],v['sampling'],v['centered'])==(s['layer'],sampling,centered));total=np.square(sv).sum();er=(np.square(hh)*cn).sum()/total;rank=sv.sum()**2/total
   errors.extend([abs(er-v['channel_net_energy_ratio']),abs(rank-v['nuclear_effective_rank'])]);spectral.append(float(np.linalg.norm(sv-np.array(v['singular_values']))/np.linalg.norm(sv)))
   cumulative=np.cumsum(np.square(sv))/total;assert int(np.searchsorted(cumulative,.9))+1==v['r90'];assert int(np.searchsorted(cumulative,.99))+1==v['r99'];assert er>=rank/4608-1e-6
assert max(errors)<1e-5 and max(spectral)<1e-6
out=dict(status='passed',independent_spectra=12,prefix_identity_checked=True,max_scalar_error=max(errors),max_spectral_relative_error=max(spectral),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'empirical_spectra_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
