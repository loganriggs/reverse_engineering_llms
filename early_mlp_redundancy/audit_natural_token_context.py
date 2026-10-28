"""Independent Torch64 replay of M4 NumPy partitions and document-half predictions."""
import json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(4);rp=P/'natural_token_context.json';r=json.loads(rp.read_text());cp=P/'natural_token_context_cohort.json';c=json.loads(cp.read_text());assert sha(cp)==r['cohort_sha256'];cache=Path(r['cache']);assert sha(cache)==r['cache_sha256'];s=torch.load(cache,weights_only=True);tokens=s['tokens'];docs=s['documents'];unique=s['unique_indices'];assert tokens.tolist()==[t for row in c['token_ids'] for t in row]
assert len(set(x['document_hash'] for x in c['records']))==64
m=json.loads(Path(c['source_manifest']).read_text());assert sha(Path(c['source_manifest']))==c['source_manifest_sha256'];reserved={x['document_hash'] for x in m['records'] if x['split']!='train'};assert not reserved.intersection(x['document_hash'] for x in c['records'])
seen=set();indices=[]
for d,row in enumerate(c['token_ids']):
 for p in range(len(row)):
  key=tuple(row[:p+1])
  if key not in seen:seen.add(key);indices.append(d*128+p)
assert indices==unique.tolist();types,counts=torch.unique(tokens[unique],return_counts=True);repeated=types[counts>=2];errors=[]
def check(a,b):
 e=abs(float(a)-b)/max(abs(b),1e-10);errors.append(e);assert e<1e-10,(a,b,e)
for row in r['rows']:
 y=s['Y'][row['layer']].double();idx=torch.arange(len(tokens)) if row['sampling']=='occurrences' else unique
 if row['subset']=='repeated_context_tokens':idx=idx[torch.isin(tokens[idx],repeated)]
 z=y[idx];tt=tokens[idx];_,inv,ns=torch.unique(tt,return_inverse=True,return_counts=True);sums=torch.zeros(len(ns),z.shape[1],dtype=torch.float64).index_add_(0,inv,z);raw=z.square().sum();group=sums.square().sum(1).div(ns).sum();global_e=z.sum(0).square().sum()/len(z);total=raw-global_e;within=raw-group;between=group-global_e
 assert len(idx)==row['rows'];check(total,row['centered_energy']);check(within,row['within_energy']);check(between,row['between_energy']);check(within/total,row['within_fraction']);check(within/raw,row['within_over_raw'])
for row in r['cross_document_prediction']:
 y=s['Y'][row['layer']].double();tr=unique[docs[unique]//32==row['train_half']];te=unique[docs[unique]//32!=row['train_half']];types,inv,ns=torch.unique(tokens[tr],return_inverse=True,return_counts=True);mu=torch.zeros(len(types),y.shape[1],dtype=torch.float64).index_add_(0,inv,y[tr])/ns[:,None];covered=te[torch.isin(tokens[te],types)];pred=mu[torch.searchsorted(types,tokens[covered])];err=(y[covered]-pred).square().sum();baseline=(y[covered]-y[tr].mean(0)).square().sum();assert len(covered)==row['covered_rows'];check(err,row['squared_error']);check(baseline,row['global_baseline_error']);check(err/baseline,row['relative_to_global_baseline']);check(err/y[covered].square().sum(),row['relative_to_raw'])
out=dict(status='passed',partitions=len(r['rows']),cross_document_predictions=len(r['cross_document_prediction']),max_relative_error=max(errors),result_sha256=sha(rp),cohort_and_cache_hashes_checked=True,reserved_BSF_document_hashes_excluded=True,unique_prefix_indices_rebuilt=True,scope='Independent statistic replay from cached writes; not a second implementation of full model.')
(P/'natural_token_context_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
