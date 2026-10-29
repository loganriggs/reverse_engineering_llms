"""M62 complete natural inventory plus independent fixed full-vector samples."""
from pathlib import Path
import json,hashlib,numpy as np,torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(z):
 z=np.asarray(z,dtype=np.float64);z=z-z.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
def main():
 torch.set_num_threads(1);rp=V/'least_squares_natural.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(P/'evaluate_least_squares_natural.py');assert r['protocol_sha256']==sha(V/'balanced_response_evaluation_protocol.json');assert r['models_sha256']==sha(V/'least_squares_evaluation_models.json');maxerr=0;checked=0
 def check(a,b):
  nonlocal maxerr
  assert np.isclose(a,b,rtol=1e-7,atol=1e-9),(a,b);maxerr=max(maxerr,abs(a-b))
 for f in r['files']:
  path=Path(f['path']);assert sha(path)==f['sha256'];s=torch.load(path,weights_only=True);p=lp(s['native']);ids=s['ids'].numpy();nativeCE=float(-np.take_along_axis(p[:,:-1],ids[:,1:,None],axis=-1).mean());w=s['write'].double().numpy()
  for c in s['conditions']:
   row=r['rows'][c['row']];assert row['document']==s['document'];q=lp(c['output']);CE=float(-np.take_along_axis(q[:,:-1],ids[:,1:,None],axis=-1).mean());check(float((np.exp(p)*(p-q)).sum(-1).mean()),row['mean_KL']);check(CE,row['CE']);check(nativeCE,row['native_CE']);check(CE-nativeCE,row['delta_CE']);check(float((s['native'].numpy().argmax(-1)==c['output'].numpy().argmax(-1)).mean()),row['top1_agreement']);check(float(np.sum((w-c['reconstruction'].double().numpy())**2)/np.sum(w**2)),row['local_relative_error']);check(float(c['support'].numpy().sum(-1).mean()),row['active_mean']);check(float((c['code'].numpy()!=0).sum(-1).mean()),row['numerical_nonzero_mean']);checked+=1
 for s in r['summaries']:
  rs=[x for x in r['rows'] if all(x[k]==s[k] for k in ['kind','seed','arm'])];assert len(rs)==32 and {x['document'] for x in rs}==set(range(32,64))
  for field in ['mean_KL','native_CE','CE','delta_CE','top1_agreement','local_relative_error','active_mean']:check(float(np.mean([x[field] for x in rs])),s[field])
 assert checked==108 and len(r['rows'])==1728 and r['forward_count']==1760;out=dict(status='passed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),inventory_rows=1728,sampled_full_vectors=108,summaries=54,max_metric_absolute_error=maxerr,scope='All model/document inventory and aggregates; independent full-output NumPy scoring of fixed docs32/63 for all54encoders; not every natural output rescored.')
 op=V/'least_squares_natural_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
