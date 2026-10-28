"""Independent NumPy audit of both document halves, all six M5 conditions."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation';torch.set_num_threads(4)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 x=x.astype(np.float64);z=x-x.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
rp=P/'token_table_causal.json';r=json.loads(rp.read_text());sp=Path(r['audit_samples']);assert sha(sp)==r['audit_samples_sha256'];assert sha(P/'token_table_causal_protocol.json')==r['protocol_sha256'];samples=torch.load(sp,weights_only=True);m4=json.loads((P/'natural_token_context.json').read_text());s=torch.load(m4['cache'],weights_only=True);t=s['tokens'].numpy();d=s['documents'].numpy();u=s['unique_indices'].numpy();errors=[];local_errors=[];table_errors=[];local_metric_errors=[]
for sample in samples:
 doc=sample['document'];ref=sample['native'].numpy();lp=logsoft(ref);ids=sample['ids'].numpy();tr=u[d[u]//32!=doc//32];native_ce=float(-np.take_along_axis(lp[:,:-1],ids[:,1:,None],axis=-1).mean())
 for c in sample['conditions']:
  row=next(x for x in r['records'] if x['document']==doc and x['layer']==c['layer'] and x['kind']==c['kind']);out=c['output'].numpy();lq=logsoft(out);kl=float((np.exp(lp)*(lp-lq)).sum(-1).mean());ce=float(-np.take_along_axis(lq[:,:-1],ids[:,1:,None],axis=-1).mean());acc=float((ref.argmax(-1)==out.argmax(-1)).mean())
  for name,val in [('KL',kl),('CE',ce),('native_CE',native_ce),('delta_CE',ce-native_ce),('top1_agreement',acc)]:
   err=abs(val-row[name]);errors.append(err);assert err<1e-8,(name,err)
  mask=c['mask'].numpy();assert np.array_equal(mask,np.isin(ids,t[tr]));assert int(mask.sum())==row['covered_positions'];y=s['Y'][c['layer']].numpy();means={int(token):y[tr[t[tr]==token]].astype(np.float64).mean(0).astype(np.float32) for token in np.unique(t[tr])};pred=c['table_product'].numpy();rebuilt=np.stack([means.get(int(token),np.zeros(1152,dtype=np.float32)) for token in ids[0]])[None];table_errors.append(float(np.max(np.abs(pred-rebuilt))));assert np.array_equal(pred,rebuilt)
  native=c['native_write'].numpy();v=c['native_product'].numpy().astype(np.float64);bias=c['bias'].numpy();proposal=pred if c['kind']=='table' else v*(np.linalg.norm(pred.astype(np.float64),axis=-1,keepdims=True)/np.linalg.norm(v,axis=-1,keepdims=True));expected=np.where(mask[:,:,None],proposal+bias,native);effective=c['effective_write'].numpy();relative=float(np.linalg.norm(effective-expected)/np.linalg.norm(expected));local_errors.append(relative);assert relative<1e-6;assert np.array_equal(effective[~mask],native[~mask]);le=float(np.square(effective.astype(np.float64)-native).sum()/np.square(v).sum());local_metric_errors.append(abs(le-row['local_relative_squared_error']));assert local_metric_errors[-1]<1e-7
# Aggregate means are separately rebuilt over all384 saved records.
for summary in r['summaries']:
 rows=[x for x in r['records'] if x['layer']==summary['layer'] and x['kind']==summary['kind']];assert len(rows)==64
 for key in ['KL','top1_agreement','CE','native_CE','delta_CE','local_relative_squared_error','covered_positions']:assert abs(np.mean([x[key] for x in rows])-summary[key])<1e-10
out=dict(status='passed',independent_full_output_checks=12,documents=[x['document'] for x in samples],max_absolute_metric_error=max(errors),max_relative_intervention_error=max(local_errors),max_table_mean_error=max(table_errors),max_absolute_local_metric_error=max(local_metric_errors),local_metric_precision_note='Production subtracts float32 writes before float64 accumulation; independent audit subtracts in float64, tolerance1e-7.',all_384_record_aggregates_checked=True,result_sha256=sha(rp),scope='NumPy output scoring, opposite-half table fitting and local intervention reconstruction for12conditions across2documents. Native tracing relies on earlier model validation; remaining documents have production invariants and aggregate checks.')
(P/'token_table_causal_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
