"""M32 independent NumPy full-vector replay and decomposition audit."""
from pathlib import Path
import hashlib,json,math,statistics
import numpy as np,torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
def lp(x):
 x=x.astype(float);x-=x.max();return x-np.log(np.exp(x).sum())
def lse(x):
 x=x.astype(float);m=x.max();return m+np.log(np.exp(x-m).sum())
rp=P/'native_response_geometry.json';r=json.loads(rp.read_text());pp=P/'native_geometry_protocol.json';p=json.loads(pp.read_text());assert r['protocol_sha256']==sha(pp);parent=json.loads((P/'wrapper_crossover_protocol.json').read_text());assert p['rows']==parent['rows'];wp=P/'wrapper_crossover_results.json';assert p['fixed_results_sha256']==r['fixed_results_sha256']==sha(wp);old=json.loads(wp.read_text());assert json.loads((P/'wrapper_crossover_audit.json').read_text())['result_sha256']==sha(wp);assert r['checkpoint_sha256']==old['checkpoint_sha256'];assert p['targets_sha256']==sha(P/'response_training_targets.json');assert p['expansion_sha256']==sha(P/'context_expansion_protocol.json');prior={(x['target_index'],x['context']):x for x in old['records'] if x['arm']=='initial'};vp=Path(r['vectors_path']);assert sha(vp)==r['vectors_sha256'];v=torch.load(vp,weights_only=True);lookup={(x['target_index'],x['context']):x for x in v['vectors']};assert len(lookup)==len(v['vectors'])==720;specs={(x['target_index'],x['context']):x for x in p['rows']};assert set(lookup)==set(specs);replays={(x['target_index'],x['context']):x for x in r['replays']};errors=[];response={}
for key,x in lookup.items():
 s=specs[key];assert x['native_key']==(s['pair_id'],s['endpoint']);native=v['nativecache'][x['native_key']].numpy();changed=x['changed'].numpy();a,b=lp(native),lp(changed);dr=b-a;response[key]=(a,dr,changed.astype(float)-native.astype(float));row=replays[key];oldrow=prior[key];delta=v['deltas'][x['target_index']].numpy();assert hashlib.sha256(delta.tobytes()).hexdigest()==row['task_delta_sha256']==oldrow['task_delta_sha256'];close(float(np.sum(delta.astype(float)**2)),oldrow['delta_energy']);assert row['noop']==oldrow['noop']==bool(np.all(delta==0));groups=x['groups'];variant=x['variant'];margin=lambda z:lse(z[groups[variant]])-lse(z[groups[1-variant]]);effect=margin(native)-margin(changed);energy=float(np.sum(np.exp(a)*dr**2));binary=margin(native)>0;top1=int(native.argmax()) in groups[variant]
 for field,val in [('native_margin_effect',effect),('native_response_energy',energy)]:close(row[field],val);close(val,oldrow[field]);errors.append(abs(val-row[field]))
 assert row['native_binary']==oldrow['native_binary']==binary and row['native_top1']==oldrow['native_top1']==top1
 if row['noop']:assert np.array_equal(native,changed)
assert set(replays)==set(specs);assert len(v['nativecache'])==r['unique_native_inputs'];assert r['forward_count']==len(v['nativecache'])+sum(not x['noop'] for x in r['replays'])
seen=set();identitymax=0;decompositionmax=0
for x in r['comparisons']:
 i=x['target_index'];c=x['context'];assert (i,c) not in seen;seen.add((i,c));a,ref,_=response[i,'reference'];_,current,logits=response[i,c];w=np.exp(a);close(float(w.sum()),1);rc=ref-np.sum(w*ref);cc=current-np.sum(w*current);lc=logits-np.sum(w*logits);identity=float(np.max(np.abs(cc-lc)));identitymax=max(identitymax,identity);assert identity<1e-10;diff=current-ref;mean=float(np.sum(w*diff));center=diff-mean;total=float(np.sum(w*diff**2));shape=float(np.sum(w*center**2));constant=mean**2;decompositionmax=max(decompositionmax,abs(total-shape-constant));close(total,shape+constant);re=float(np.sum(w*ref**2));rce=float(np.sum(w*rc**2));cce=float(np.sum(w*cc**2));values=dict(total_difference=total,centered_difference=shape,constant_difference=constant,reference_response_energy=re,reference_centered_energy=rce,context_centered_energy=cce)
 for k,val in values.items():close(val,x[k]);errors.append(abs(val-x[k]))
 if rce and cce:close(float(np.sum(w*rc*cc)/np.sqrt(rce*cce)),x['centered_cosine'])
 else:assert x['centered_cosine'] is None
 if rce:close(float(np.sqrt(cce/rce)),x['centered_norm_ratio'])
 else:assert x['centered_norm_ratio'] is None
assert seen=={(s['target_index'],c) for s in p['rows'] for c in ['passage','sample']} and len(seen)==480
for s in r['summaries']:
 rows=[x for x in r['comparisons'] if all(x[k]==s[k] for k in ['kind','seed','context']) and (s['family'] is None or x['family']==s['family'])];assert len(rows)==s['directions'];valid=[x for x in rows if x['centered_cosine'] is not None];assert len(valid)==s['nonzero_geometry']
 for k,val in s['sums'].items():close(sum(x[k] for x in rows),val)
 for key,num,den in [('centered_fraction','centered_difference','total_difference'),('relative_total_difference','total_difference','reference_response_energy'),('relative_centered_difference','centered_difference','reference_centered_energy')]:
  if s['sums'][den]:close(s[key],s['sums'][num]/s['sums'][den])
  else:assert s[key] is None
 for key,field in [('median_centered_cosine','centered_cosine'),('median_centered_norm_ratio','centered_norm_ratio')]:
  if valid:close(s[key],statistics.median(x[field] for x in valid))
  else:assert s[key] is None
out=dict(status='passed',result_sha256=sha(rp),full_native_vector_replays=720,paired_decompositions=480,summaries=len(r['summaries']),max_metric_error=max(errors),max_centered_identity_error=identitymax,max_additive_identity_error=decompositionmax,scope='All cached native response vectors independently rescored in NumPy, old effects/competence/delta hashes, common metric, centered-logit/logprob identity, additive decomposition, full coverage and aggregates; native transformer not independently implemented.');(P/'native_response_geometry_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
