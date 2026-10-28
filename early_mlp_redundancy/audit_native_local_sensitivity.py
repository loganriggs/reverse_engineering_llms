"""Independent Torch scoring of M33 NumPy analysis and complete inventory."""
from pathlib import Path
import hashlib,json,math,statistics
import torch
P=Path(__file__).resolve().parent/'validation'
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
errors=[]
def close(a,b):
 if a is None or b is None:assert a is b;return
 assert math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-9),(a,b)
 errors.append(abs(a-b))
def energy(x,w):return float((w*x.square()).sum())
def center(x,w):return x-(w*x).sum()
def geometry(u,v,w):
 er,ec=energy(u,w),energy(v,w);dot=float((w*u*v).sum());residual=v-(dot/er)*u if er else v
 return dict(reference_energy=er,context_energy=ec,difference_energy=energy(v-u,w),cosine=dot/math.sqrt(er*ec) if er and ec else None,orthogonal_energy=energy(residual,w))
rp=P/'native_local_results.json';r=json.loads(rp.read_text());ip=P/'native_local_inventory.json';inv=json.loads(ip.read_text());pp=P/'native_local_protocol.json';p=json.loads(pp.read_text());assert r['inventory_sha256']==sha(ip);assert r['protocol_sha256']==inv['protocol_sha256']==sha(pp)
assert inv['source_sha256']==sha(P.parent/'probe_native_local_sensitivity.py')
parent=json.loads((P/'native_response_geometry.json').read_text());assert inv['parent_result_sha256']==p['parent_result_sha256']==sha(P/'native_response_geometry.json');assert inv['checkpoint_sha256']==parent['checkpoint_sha256']
assert p['rows']==json.loads((P/'native_geometry_protocol.json').read_text())['rows']
assert p['expansion_sha256']==sha(P/'context_expansion_protocol.json')
for result in [inv,parent]:assert sha(Path(result['vectors_path']))==result['vectors_sha256']
v=torch.load(inv['vectors_path'],weights_only=True);f=torch.load(parent['vectors_path'],weights_only=True)
assert v['protocol_sha256']==sha(pp)
assert len(v['records'])==inv['records']==720
assert len(v['baselines'])==inv['unique_inputs']==218
for k,t in v['baselines'].items():assert torch.equal(t,f['nativecache'][k])
lookup={(x['target_index'],x['context']):x for x in v['records']};fl={(x['target_index'],x['context']):x for x in f['vectors']};specs={(x['target_index'],x['context']):x for x in p['rows']};rows={(x['target_index'],x['context']):x for x in r['rows']}
assert len(lookup)==len(rows)==len(specs)==720 and lookup.keys()==rows.keys()==specs.keys()
responses={};weights={};counts=dict(rows=720,noops=0,scale_consistent=0,resolved=0,usable=0)
for key,x in lookup.items():
 i,c=key;s=specs[key];o=rows[key]
 assert x['native_key']==(s['pair_id'],s['endpoint'])
 for k in ['kind','seed','family']:assert x[k]==o[k]==s[k]
 delta=f['deltas'][i]
 assert x['task_delta_sha256']==hashlib.sha256(delta.numpy().tobytes()).hexdigest()
 noop=not bool(torch.count_nonzero(delta));assert x['noop']==o['noop']==noop;counts['noops']+=noop
 base=v['baselines'][lookup[i,'reference']['native_key']].double();w=base.softmax(-1);weights[i]=w;ds={};resolved=True
 assert set(x['vectors'])=={'0.1:-1','0.1:1','0.01:-1','0.01:1'}
 for h in p['amplitudes']:
  a,b=x['vectors'][f'{h}:1'],x['vectors'][f'{h}:-1'];assert torch.isfinite(a).all() and torch.isfinite(b).all()
  if noop:assert torch.equal(a,v['baselines'][x['native_key']]) and torch.equal(a,b)
  raw=a.double()-b.double();ds[str(h)]=center(raw/(2*h),w)
  ulps=sum((torch.nextafter(z,torch.copysign(torch.full_like(z,float('inf')),z))-z).abs().double() for z in [a,b])
  resolution=math.sqrt(energy(raw,w)/energy(ulps,w));close(resolution,o['resolution_ratios'][str(h)]);resolved &= resolution>=p['resolution_ratio_threshold']
 fv=fl[key];ds['full']=center(fv['changed'].double()-f['nativecache'][fv['native_key']].double(),w)
 e=energy(ds['0.01'],w);change=energy(ds['0.01']-ds['0.1'],w);relative=math.sqrt(change/e) if e else None
 close(e,o['local_energy']);close(change,o['scale_difference_energy']);close(relative,o['scale_relative_difference']);consistent=relative is not None and relative<=p['convergence_relative_norm_threshold']
 for k,val in [('scale_consistent',consistent),('resolved',resolved),('usable',consistent and resolved)]:assert o[k]==val;counts[k]+=val
 for k,val in geometry(ds['0.01'],ds['full'],w).items():close(val,o['full_vs_local'][k])
 responses[key]=ds
assert counts==r['counts'];assert inv['noops']==counts['noops']==93
assert v['calls']==inv['forward_count']==218+4*(720-93)==2726
seen=set()
for x in r['pairs']:
 i,c,h=x['target_index'],x['context'],x['amplitude'];key=(i,c,h);assert key not in seen;seen.add(key)
 assert x['jointly_usable']==(rows[i,'reference']['usable'] and rows[i,c]['usable'])
 for k in ['kind','seed','family']:assert x[k]==rows[i,c][k]
 for k,val in geometry(responses[i,'reference'][h],responses[i,c][h],weights[i]).items():close(val,x[k])
 if h=='full':
  old=next(y for y in parent['comparisons'] if y['target_index']==i and y['context']==c)
  close(x['difference_energy'],old['centered_difference']);close(x['reference_energy'],old['reference_centered_energy']);close(x['cosine'],old['centered_cosine'])
assert len(seen)==1440
summarykeys=set()
for s in r['summaries']:
 key=tuple(s[k] for k in ['kind','seed','context','subset','amplitude']);assert key not in summarykeys;summarykeys.add(key)
 rs=[x for x in r['pairs'] if all(x[k]==s[k] for k in ['kind','seed','context','amplitude']) and (s['subset']=='all' or x['jointly_usable'])]
 assert len(rs)==s['count'];cos=[x['cosine'] for x in rs if x['cosine'] is not None];assert len(cos)==s['nonzero']
 total=sum(x['difference_energy'] for x in rs);ref=sum(x['reference_energy'] for x in rs)
 close(total/ref if ref else None,s['relative_difference']);close(sum(x['orthogonal_energy'] for x in rs)/total if total else None,s['orthogonal_fraction']);close(statistics.median(cos) if cos else None,s['median_cosine'])
assert len(summarykeys)==48
out=dict(status='passed',result_sha256=sha(rp),inventory_sha256=sha(ip),row_checks=720,paired_geometry_checks=1440,aggregate_checks=48,baseline_replays=218,max_absolute_metric_error=max(errors),scope='Independent Torch rescoring of NumPy analysis, all output vectors, noops, ULP diagnostics, scale agreement, parent full-response identities and inventories; transformer itself not independently implemented.')
(P/'native_local_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
