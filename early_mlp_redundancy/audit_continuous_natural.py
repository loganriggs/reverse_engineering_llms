"""M112 independent NumPy audit of every saved natural output and reconstruction."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha


def main():
 torch.set_num_threads(1);rp=V/'continuous_natural.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(V.parent/'evaluate_continuous_natural.py') and r['models_sha256']==sha(V/'continuous_training_results.json')
 models=json.loads((V/'continuous_training_results.json').read_text())['models'];states={}
 for e in models:
  assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True);states[e['seed'],e['mode']]=s
 maximum=0.;decodeerr=0.;count=0;seen=set()
 def logp(x):
  x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
 for f in r['files']:
  assert sha(f['path'])==f['sha256'];b=torch.load(f['path'],weights_only=True);native=b['native'].double().numpy();lp=logp(native);ids=b['ids'].numpy();w=b['write'].double().numpy();nce=float(-np.take_along_axis(lp[:,:-1],ids[:,1:,None],-1).mean())
  for c in b['conditions']:
   row=c['row'];key=(row['seed'],row['mode'],row['document']);assert key not in seen;seen.add(key);out=c['output'].double().numpy();op=logp(out);q=c['reconstruction'].double().numpy();ce=float(-np.take_along_axis(op[:,:-1],ids[:,1:,None],-1).mean())
   metrics=dict(mean_KL=float((np.exp(lp)*(lp-op)).sum(-1).mean()),native_CE=nce,CE=ce,delta_CE=ce-nce,top1_agreement=float(np.mean(native.argmax(-1)==out.argmax(-1))),local_relative_error=float(np.square(w-q).sum()/np.square(w).sum()),active_mean=float(c['support'].double().sum(-1).mean()),code_energy=float(c['code'].double().square().mean()))
   for k,val in metrics.items():err=abs(row[k]-val);maximum=max(maximum,err);assert err<1e-9,(k,err);count+=1
   s=states[row['seed'],row['mode']];st={k:v.double().numpy() for k,v in s['state'].items()};mu=s['mean'].double().numpy();scale=s['scale'].double().numpy();a=(((w-mu)/scale)@st['encoder'].T+st['encoder_bias']).reshape(1,128,32,4);norm=np.linalg.norm(a,axis=-1)
   if row['mode']=='continuous':
    tau=np.sort(norm,axis=-1)[...,-9,None];code=a*(np.maximum(norm-tau,0)/np.where(norm>0,norm,1))[...,None]
   else:
    indices=np.argsort(norm,axis=-1)[...,-8:];mask=np.zeros_like(norm);np.put_along_axis(mask,indices,1,axis=-1);code=a*mask[...,None]
   D=st['dictionary'];D=D/np.linalg.norm(D,axis=0);decoded=code.reshape(1,128,128)@D.T*scale+st['bias']*scale+mu;err=float(np.linalg.norm(decoded-q)/max(np.linalg.norm(q),1));decodeerr=max(decodeerr,err);assert err<1e-5
 expected={(s,m,d) for s in [71,72] for m in ['hard','continuous'] for d in range(32,64)};assert seen==expected
 assert r['forwards']==160 and r['historical_hard_metric_replays']==64 and r['native_write_replays']==32
 for summary in r['summaries']:
  rr=[x for x in r['rows'] if x['seed']==summary['seed'] and x['mode']==summary['mode']];assert len(rr)==32
  for key,value in summary.items():
   if key in ['seed','mode']:continue
   assert abs(sum(x[key] for x in rr)/32-value)<1e-12
 out=dict(status='passed',source_sha256=sha(__file__),results_sha256=sha(rp),rows_checked=len(seen),metric_checks=count,max_metric_error=maximum,max_independent_decode_error=decodeerr,scope='All saved natural logits and local reconstructions audited independently in NumPy; original native writes and hard metrics replayed separately by producer. Does not test task causal transfer.')
 dest=V/'continuous_natural_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
