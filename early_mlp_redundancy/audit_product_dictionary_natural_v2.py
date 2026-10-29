"""M95 all-output retention audit with independent predictor arithmetic."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V
from independent_dictionary_predictions import IndependentDictionaryPrediction
P=V.parent

def lp(t):
 x=t.double().numpy();x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))

def main():
 torch.set_num_threads(1);rp=V/'product_dictionary_natural.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(P/'evaluate_product_dictionary_natural.py');ep=V/'product_dictionary_evaluation_execution.json';assert r['protocol_sha256']==sha(ep);mp=V/'product_dictionary_evaluation_models.json';assert r['models_sha256']==sha(mp);tol={'metric_relative':1e-7,'metric_absolute':1e-9};entries=json.loads(mp.read_text())['models'];identity=lambda e:tuple(e[k] for k in ['kind','seed','arm']);predictors={identity(e):IndependentDictionaryPrediction(e) for e in entries};cohort=json.loads((V/'natural_token_context_cohort.json').read_text());seen=set();maximum=0;maxdecode=0;maxcode=0
 def check(x,y):
  nonlocal maximum
  assert np.isclose(x,y,rtol=tol['metric_relative'],atol=tol['metric_absolute']),(x,y);maximum=max(maximum,abs(x-y))
 for file in r['files']:
  assert sha(file['path'])==file['sha256'];s=torch.load(file['path'],weights_only=True,mmap=True);n=lp(s['native']);ids=s['ids'].numpy();assert ids[0].tolist()==cohort['token_ids'][s['document']];nativece=float(-np.take_along_axis(n[:,:-1],ids[:,1:,None],axis=-1).mean());w=s['write'].double().numpy();assert len(s['conditions'])==6;products=next(iter(predictors.values())).prepare(s['input'])
  for c in s['conditions']:
   i=c['row'];assert i not in seen;seen.add(i);row=r['rows'][i];assert row['document']==s['document'];enc=predictors[identity(row)];de,ce=enc.verify(products,c['reconstruction'],c['support'],c['code']);maxdecode=max(maxdecode,de);maxcode=max(maxcode,ce);p=lp(c['output']);cross=float(-np.take_along_axis(p[:,:-1],ids[:,1:,None],axis=-1).mean());check(float((np.exp(n)*(n-p)).sum(-1).mean()),row['mean_KL']);check(cross,row['CE']);check(nativece,row['native_CE']);check(cross-nativece,row['delta_CE']);check(float((s['native'].numpy().argmax(-1)==c['output'].numpy().argmax(-1)).mean()),row['top1_agreement']);check(float(np.square(w-c['reconstruction'].double().numpy()).sum()/np.square(w).sum()),row['local_relative_error']);check(float(c['support'].double().sum(-1).mean()),row['active_mean']);check(float((c['code']!=0).double().sum(-1).mean()),row['numerical_nonzero_mean']);assert row['rank_mean'] is None;assert row['retained_native_products']==4608;assert row['output_dimension']==128
  print('AUDIT_DOC',s['document'],flush=True)
 assert seen==set(range(192)) and r['forward_count']==224 and len(r['summaries'])==6
 for summary in r['summaries']:
  rows=[x for x in r['rows'] if identity(x)==identity(summary)];assert len(rows)==32 and {x['document'] for x in rows}==set(range(32,64))
  for k in ['mean_KL','native_CE','CE','delta_CE','top1_agreement','local_relative_error','active_mean']:check(float(np.mean([x[k] for x in rows])),summary[k])
 out=dict(status='passed',source_sha256=sha(__file__),prediction_helper_sha256=sha(P/'independent_dictionary_predictions.py'),result_sha256=sha(rp),independently_scored_full_vectors=192,max_metric_absolute_error=maximum,max_prediction_relative_error=maxdecode,max_code_relative_error=maxcode,scope='All192fulloutputvectors independentlyscored. All predictions independently computed with float64 NumPy products, normalization, coefficient and decoder matrices. No independent transformer implementation.')
 dest=V/'product_dictionary_natural_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',192,maximum,maxdecode)
if __name__=='__main__':main()
