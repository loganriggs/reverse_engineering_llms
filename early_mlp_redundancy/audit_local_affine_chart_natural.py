"""M78 complete independent natural output and affine input prediction audit."""
from pathlib import Path
import json,hashlib,numpy as np,torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(z):
 z=np.asarray(z,dtype=np.float64);z=z-z.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
def main():
 torch.set_num_threads(1);rp=V/'local_affine_chart_natural.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(P/'evaluate_local_affine_chart_natural.py');assert r['protocol_sha256']==sha(V/'local_affine_chart_evaluation_execution.json');assert r['models_sha256']==sha(V/'local_affine_chart_evaluation_models.json');fit=json.loads((V/'local_affine_chart_fit.json').read_text());states={}
 for e in fit['models']:
  assert sha(Path(e['path']))==e['sha256'];states[e['name']]=np.load(e['path'])
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(Path(ref['path']))==ref['sha256'];native=torch.load(ref['path'],weights_only=True);L,R,D=[native[k].double().numpy() for k in ['L','R','D']];cohort=json.loads((V/'natural_token_context_cohort.json').read_text());maxerr=0;checked=0;seen=set()
 def check(a,b):
  nonlocal maxerr
  assert np.isclose(a,b,rtol=1e-7,atol=1e-9),(a,b);maxerr=max(maxerr,abs(a-b))
 for f in r['files']:
  path=Path(f['path']);assert sha(path)==f['sha256'];s=torch.load(path,weights_only=True);p=lp(s['native']);ids=s['ids'].numpy();nativeCE=float(-np.take_along_axis(p[:,:-1],ids[:,1:,None],axis=-1).mean());w=s['write'].double().numpy();assert ids[0].tolist()==cohort['token_ids'][s['document']]
  for c in s['conditions']:
   row=r['rows'][c['row']];assert row['document']==s['document'];assert c['row'] not in seen;seen.add(c['row']);rank=1152;code=s['input'].double().numpy();state=states[row['arm']];x=code.reshape(-1,1152);anchors=state['anchors'];labels=np.sum((x[:,None,:]-anchors[None,:,:])**2,axis=2).argmin(1);center=anchors[labels];dx=x-center;reference=(((center@L.T)*(center@R.T)+(center@L.T)*(dx@R.T)+(dx@L.T)*(center@R.T))@D.T).reshape(code.shape);assert np.linalg.norm(reference-c['reconstruction'].numpy())/max(1,np.linalg.norm(reference))<1e-5;assert np.array_equal(code,c['code'].numpy());assert bool((c['support']==1).all());check(row['rank_mean'],rank);q=lp(c['output']);CE=float(-np.take_along_axis(q[:,:-1],ids[:,1:,None],axis=-1).mean());check(float((np.exp(p)*(p-q)).sum(-1).mean()),row['mean_KL']);check(CE,row['CE']);check(nativeCE,row['native_CE']);check(CE-nativeCE,row['delta_CE']);check(float((s['native'].numpy().argmax(-1)==c['output'].numpy().argmax(-1)).mean()),row['top1_agreement']);check(float(np.sum((w-c['reconstruction'].double().numpy())**2)/np.sum(w**2)),row['local_relative_error']);check(float(c['support'].numpy().sum(-1).mean()),row['active_mean']);check(float((c['code'].numpy()!=0).sum(-1).mean()),row['numerical_nonzero_mean']);checked+=1
 for s in r['summaries']:
  rs=[x for x in r['rows'] if all(x[k]==s[k] for k in ['kind','seed','arm'])];assert len(rs)==32 and {x['document'] for x in rs}==set(range(32,64))
  for field in ['mean_KL','native_CE','CE','delta_CE','top1_agreement','local_relative_error','active_mean']:check(float(np.mean([x[field] for x in rs])),s[field])
 assert checked==96 and len(r['rows'])==96 and r['forward_count']==128 and seen==set(range(96)) and {f['document'] for f in r['files']}==set(range(32,64));out=dict(status='passed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),inventory_rows=96,independently_scored_full_vectors=96,summaries=3,max_metric_absolute_error=maxerr,scope='All96 model/document outputs independently scored with NumPy, all chart outputs independently computed with direct distances and native factored polynomial from normalized inputs, full inventory and aggregates checked.')
 op=V/'local_affine_chart_natural_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
