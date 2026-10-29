"""M73 complete independent natural output scoring and PCA reconstruction audit."""
from pathlib import Path
import json,hashlib,numpy as np,torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(z):
 z=np.asarray(z,dtype=np.float64);z=z-z.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
def main():
 torch.set_num_threads(1);rp=V/'pca_natural.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(P/'evaluate_pca_natural.py');assert r['protocol_sha256']==sha(V/'pca_encoder_baseline_protocol.json');assert r['models_sha256']==sha(V/'pca_evaluation_models.json');fit=json.loads((V/'pca_encoder_fit.json').read_text());assert sha(Path(fit['state_path']))==fit['state_sha256'];state=np.load(fit['state_path']);cohort=json.loads((V/'natural_token_context_cohort.json').read_text());maxerr=0;checked=0;seen=set()
 def check(a,b):
  nonlocal maxerr
  assert np.isclose(a,b,rtol=1e-7,atol=1e-9),(a,b);maxerr=max(maxerr,abs(a-b))
 for f in r['files']:
  path=Path(f['path']);assert sha(path)==f['sha256'];s=torch.load(path,weights_only=True);p=lp(s['native']);ids=s['ids'].numpy();nativeCE=float(-np.take_along_axis(p[:,:-1],ids[:,1:,None],axis=-1).mean());w=s['write'].double().numpy();assert ids[0].tolist()==cohort['token_ids'][s['document']]
  for c in s['conditions']:
   row=r['rows'][c['row']];assert row['document']==s['document'];assert c['row'] not in seen;seen.add(c['row']);rank=int(row['arm'].removeprefix('rank'));assert rank in (32,128);u=state['basis'][:,:rank];reference=(state['mean']+(w-state['mean'])@(u@u.T)).astype(np.float32);assert np.allclose(reference,c['reconstruction'].numpy(),rtol=1e-6,atol=1e-4);assert np.allclose((w-state['mean'])@u,c['code'].numpy(),rtol=1e-10,atol=1e-9);assert bool((c['support']==1).all());check(row['rank_mean'],rank);q=lp(c['output']);CE=float(-np.take_along_axis(q[:,:-1],ids[:,1:,None],axis=-1).mean());check(float((np.exp(p)*(p-q)).sum(-1).mean()),row['mean_KL']);check(CE,row['CE']);check(nativeCE,row['native_CE']);check(CE-nativeCE,row['delta_CE']);check(float((s['native'].numpy().argmax(-1)==c['output'].numpy().argmax(-1)).mean()),row['top1_agreement']);check(float(np.sum((w-c['reconstruction'].double().numpy())**2)/np.sum(w**2)),row['local_relative_error']);check(float(c['support'].numpy().sum(-1).mean()),row['active_mean']);check(float((c['code'].numpy()!=0).sum(-1).mean()),row['numerical_nonzero_mean']);checked+=1
 for s in r['summaries']:
  rs=[x for x in r['rows'] if all(x[k]==s[k] for k in ['kind','seed','arm'])];assert len(rs)==32 and {x['document'] for x in rs}==set(range(32,64))
  for field in ['mean_KL','native_CE','CE','delta_CE','top1_agreement','local_relative_error','active_mean']:check(float(np.mean([x[field] for x in rs])),s[field])
 assert checked==64 and len(r['rows'])==64 and r['forward_count']==96 and seen==set(range(64)) and {f['document'] for f in r['files']}==set(range(32,64));out=dict(status='passed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),inventory_rows=64,independently_scored_full_vectors=64,summaries=2,max_metric_absolute_error=maxerr,scope='All64 model/document outputs independently scored with NumPy, all PCA reconstructions checked using explicit projectors, full inventory and aggregates checked.')
 op=V/'pca_natural_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
