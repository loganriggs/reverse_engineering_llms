"""Independent Torch group-sum and spectral audit of NumPy token/context partition."""
import json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent/'validation';path=P/'token_context_partition.json';r=json.loads(path.read_text());m=json.loads((P/'empirical_spectra.json').read_text());saved=torch.load('/tmp/emlp_m2_matrices.pt',weights_only=True);torch.set_num_threads(4)
tokens=torch.tensor([m['prefixes'][o['prefix_id']][-1] for o in m['occurrences']]);ut=torch.tensor([p[-1] for p in m['prefixes']]);counts={int(t):int((ut==t).sum()) for t in ut.unique()};errors=[];specerrors=[]
assert len(counts)==r['token_types']==123 and sum(c==1 for c in counts.values())==r['single_prefix_token_types']==53
for sample in saved:
 Y=sample['Y'].double()
 for view in ('occurrences','unique_prefixes'):
  ix=torch.arange(len(Y)) if view=='occurrences' else torch.tensor(m['unique_indices']);y=Y[ix];tok=tokens[ix]
  for subset in ('all_tokens','repeated_context_tokens'):
   use=torch.ones(len(y),dtype=torch.bool) if subset=='all_tokens' else torch.tensor([counts[int(t)]>=2 for t in tok]);z=y[use];tt=tok[use];types,inverse,nn=torch.unique(tt,return_inverse=True,return_counts=True);sums=torch.zeros(len(types),z.shape[1],dtype=torch.float64);sums.index_add_(0,inverse,z);mu=sums/nn[:,None];means=mu[inverse];within=z-means;between=means-z.mean(0);rec=next(x for x in r['rows'] if (x['layer'],x['sampling'],x['subset'])==(sample['layer'],view,subset));assert len(z)==rec['rows'] and len(types)==rec['token_types']
   total=z.square().sum()-len(z)*z.mean(0).square().sum();we=z.square().sum()-(nn[:,None]*mu.square()).sum();be=(nn[:,None]*(mu-z.mean(0)).square()).sum()
   for value,key in [(total,'centered_energy'),(we,'within_token_energy'),(be,'between_token_energy')]:errors.append(abs(float(value)-rec[key])/max(rec['centered_energy'],1))
   assert abs(float(we/total)-rec['within_fraction'])<1e-10
   for group in rec['groups']:
    t=group['token'];assert group['distinct_prefixes']==counts[t] and group['rows']==int((tt==t).sum())
   if subset=='all_tokens':
    for key,mat in [('within',within),('between',between)]:
     sv=torch.linalg.svdvals(mat);recorded=torch.tensor(rec['spectra'][key]['singular_values'],dtype=torch.float64);specerrors.append(float((sv-recorded).norm()/sv.norm()))
assert max(errors)<1e-10 and max(specerrors)<1e-8
out=dict(status='passed',partitions=12,independent_spectra=12,max_relative_energy_error=max(errors),max_relative_spectral_error=max(specerrors),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'token_context_partition_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
