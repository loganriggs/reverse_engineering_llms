"""M88 all-output retention audit with independent predictor arithmetic."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V
from independent_bilinear_predictions import IndependentBilinearPrediction
P=V.parent

def lp(t):
 x=t.double().numpy();x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))

def main():
 torch.set_num_threads(1);rp=V/'bilinear_input_natural.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(P/'evaluate_bilinear_input_natural.py');ep=V/'bilinear_input_evaluation_execution.json';assert r['protocol_sha256']==sha(ep);mp=V/'bilinear_input_evaluation_models.json';assert r['models_sha256']==sha(mp);tol=json.loads(ep.read_text())['audit_tolerances'];entries=json.loads(mp.read_text())['models'];identity=lambda e:tuple(e[k] for k in ['kind','seed','arm']);predictors={identity(e):IndependentBilinearPrediction(e,tol) for e in entries};cohort=json.loads((V/'natural_token_context_cohort.json').read_text());seen=set();maximum=0;maxdecode=0;maxcode=0
 def check(x,y):
  nonlocal maximum
  assert np.isclose(x,y,rtol=tol['metric_relative'],atol=tol['metric_absolute']),(x,y);maximum=max(maximum,abs(x-y))
 for file in r['files']:
  assert sha(file['path'])==file['sha256'];s=torch.load(file['path'],weights_only=True);n=lp(s['native']);ids=s['ids'].numpy();assert ids[0].tolist()==cohort['token_ids'][s['document']];nativece=float(-np.take_along_axis(n[:,:-1],ids[:,1:,None],axis=-1).mean());w=s['write'].double().numpy();assert len(s['conditions'])==32
  for c in s['conditions']:
   i=c['row'];assert i not in seen;seen.add(i);row=r['rows'][i];assert row['document']==s['document'];enc=predictors[identity(row)];de,ce=enc.verify(s['input'],s['write'],c['reconstruction'],c['support'],c['code']);maxdecode=max(maxdecode,de);maxcode=max(maxcode,ce);p=lp(c['output']);cross=float(-np.take_along_axis(p[:,:-1],ids[:,1:,None],axis=-1).mean());check(float((np.exp(n)*(n-p)).sum(-1).mean()),row['mean_KL']);check(cross,row['CE']);check(nativece,row['native_CE']);check(cross-nativece,row['delta_CE']);check(float((s['native'].numpy().argmax(-1)==c['output'].numpy().argmax(-1)).mean()),row['top1_agreement']);check(float(np.square(w-c['reconstruction'].double().numpy()).sum()/np.square(w).sum()),row['local_relative_error']);check(float(c['support'].double().sum(-1).mean()),row['active_mean']);check(float((c['code']!=0).double().sum(-1).mean()),row['numerical_nonzero_mean']);assert row['rank_mean']==(128.0 if enc.variant=='output_oracle' else None)
  print('AUDIT_DOC',s['document'],flush=True)
 assert seen==set(range(1024)) and r['forward_count']==1056 and len(r['summaries'])==32
 for summary in r['summaries']:
  rows=[x for x in r['rows'] if identity(x)==identity(summary)];assert len(rows)==32 and {x['document'] for x in rows}==set(range(32,64))
  for k in ['mean_KL','native_CE','CE','delta_CE','top1_agreement','local_relative_error','active_mean']:check(float(np.mean([x[k] for x in rows])),summary[k])
 gates=[dict(kind=k[0],seed=k[1],arm=k[2],ambiguous_support_rows=e.ambiguous,min_score_gap=e.minimum_gap if np.isfinite(e.minimum_gap) else None) for k,e in predictors.items()]
 out=dict(status='passed',source_sha256=sha(__file__),prediction_helper_sha256=sha(P/'independent_bilinear_predictions.py'),result_sha256=sha(rp),independently_scored_full_vectors=1024,max_metric_absolute_error=maximum,max_prediction_relative_error=maxdecode,max_code_relative_error=maxcode,gate_audits=gates,scope='All1024fulloutputvectors independentlyscored. Inputpredictions independentlycomputed frombilinearparameters andverifiedsupport; oraclefromQR ratherthanproducerSVD. Numerical-boundary support differences reported explicitly. No independenttransformerimplementation.')
 dest=V/'bilinear_input_natural_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',1024,maximum,maxdecode,sum(x['ambiguous_support_rows'] for x in gates))
if __name__=='__main__':main()
