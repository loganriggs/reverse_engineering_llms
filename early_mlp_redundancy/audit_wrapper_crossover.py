"""M30 independent pairing, fixed-delta mapping, scores and complete coverage."""
from pathlib import Path
import hashlib,json,math
import numpy as np,torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
def lp(x):
 x=x.astype(float);x-=x.max();return x-np.log(np.exp(x).sum())
def lse(x):
 x=x.astype(float);m=x.max();return m+np.log(np.exp(x-m).sum())
pp=P/'wrapper_crossover_protocol.json';p=json.loads(pp.read_text());rp=P/'wrapper_crossover_results.json';r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(pp);up=P/'unseen_direction_protocol.json';assert p['unseen_protocol_sha256']==sha(up);u=json.loads(up.read_text());assert p['models']==u['models'];tp=P/'response_training_targets.json';assert p['targets_sha256']==sha(tp);t=json.loads(tp.read_text());ep=P/'context_expansion_protocol.json';assert p['expansion_sha256']==sha(ep);expanded={x['id']:x for x in json.loads(ep.read_text())['rows']};refp=P/'unseen_direction_results.json';ref=json.loads(refp.read_text());assert r['reference_results_sha256']==sha(refp);assert json.loads((P/'unseen_direction_audit.json').read_text())['result_sha256']==sha(refp);refs={(x['target_index'],x['arm']):x for x in ref['records']};sp=Path(ref['samples_path']);assert sha(sp)==ref['samples_sha256'];oldsamples=torch.load(sp,weights_only=True);oldsample={(ref['records'][s['row']]['target_index'],ref['records'][s['row']]['arm']):s for s in oldsamples};protocolkeys=set()
for s in p['rows']:
 i=s['target_index'];prior=u['rows'][i];assert prior['region']=='task';assert all(s[k]==prior[k] for k in ['kind','seed','family','group']);source=expanded[s['source_pair_id']];dest=expanded[s['pair_id']];ep=s['endpoint'];assert source['id']==t['records'][i]['pair_id'] and ep==t['records'][i]['endpoint'];assert source['base_id']==dest['base_id'];assert source['lexical_split']==dest['lexical_split']=='train';assert s['prefix_length']==dest['prefix_length'] and s['source_prefix_length']==source['prefix_length'];assert s['task_token_ids']==source['token_ids'][ep][source['prefix_length']:]==dest['token_ids'][ep][dest['prefix_length']:];assert source['answer_groups'][ep]==dest['answer_groups'][ep] and source['variants'][ep]==dest['variants'][ep]
 if s['context']=='reference':assert source==dest
 else:assert s['context']==dest['template'] and dest['template_split']=='evaluation'
 key=i,s['context'];assert key not in protocolkeys;protocolkeys.add(key)
assert protocolkeys=={(x['target_index'],c) for x in u['rows'] if x['region']=='task' for c in p['contexts']};assert len(protocolkeys)==720
for e in p['models']:assert sha(Path(e['path']))==e['sha256']
sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);errors=[]
for s in samples:
 x=r['records'][s['row']];spec=next(z for z in p['rows'] if (z['target_index'],z['context'])==(x['target_index'],x['context']));original=oldsample[x['target_index'],x['arm']]['delta'].numpy();task=original[:,spec['source_prefix_length']:];np.testing.assert_array_equal(task,s['task_delta'].numpy());delta=s['delta'].numpy();assert np.all(delta[:,:spec['prefix_length']]==0);np.testing.assert_array_equal(delta[:,spec['prefix_length']:],task);assert s['ids'][0,spec['prefix_length']:].tolist()==spec['task_token_ids'];assert hashlib.sha256(task.tobytes()).hexdigest()==x['task_delta_sha256'];close(float(np.sum(delta.astype(float)**2)),x['delta_energy'])
 n,nc,q,qc=[s[k].numpy() for k in ['native','native_changed','intact','changed']];a,b,c,d=map(lp,[n,nc,q,qc]);nr=b-a;fr=d-c;R=float(np.sum(np.exp(a)*(fr-nr)**2));energy=float(np.sum(np.exp(a)*nr**2));g=s['groups'];v=s['variant'];margin=lambda x:lse(x[g[v]])-lse(x[g[1-v]]);nm=margin(n)-margin(nc);qm=margin(q)-margin(qc)
 for field,val in [('R',R),('native_response_energy',energy),('native_margin_effect',nm),('reconstructed_margin_effect',qm),('margin_squared_error',(qm-nm)**2),('intact_KL',float(np.sum(np.exp(a)*(a-c))))]:close(x[field],val);errors.append(abs(x[field]-val))
 assert x['native_binary']==(margin(n)>0) and x['native_top1']==(int(n.argmax()) in g[v]);assert x['reconstructed_binary']==(margin(q)>0) and x['reconstructed_top1']==(int(q.argmax()) in g[v])
keys=set();fixed={};directions={};specs={(x['target_index'],x['context']):x for x in p['rows']};reference_count=0
for x in r['records']:
 i=x['target_index'];s=specs[i,x['context']];key=i,x['context'],x['arm'];assert key not in keys;keys.add(key);assert all(x[k]==s[k] for k in ['kind','seed','family','group','pair_id','endpoint','prefix_length']);fixedvalue=x['task_delta_sha256'],x['delta_energy'],x['noop'];assert i not in fixed or (fixed[i][0]==fixedvalue[0] and fixed[i][2]==fixedvalue[2]);close(fixed.get(i,fixedvalue)[1],fixedvalue[1]);fixed[i]=fixedvalue;dk=i,x['context'];value=tuple(x[k] for k in ['noop','native_response_energy','native_margin_effect','native_margin_energy','native_binary','native_top1']);assert dk not in directions or directions[dk]==value;directions[dk]=value
 if x['noop']:assert all(x[k]==0 for k in ['R','delta_energy','native_response_energy','margin_squared_error','native_margin_effect','reconstructed_margin_effect'])
 if x['context']=='reference':
  prior=refs[i,x['arm']];reference_count+=1
  for field in ['R','native_response_energy','native_margin_effect','reconstructed_margin_effect','delta_energy']:close(x[field],prior[field])
assert keys=={(s['target_index'],s['context'],e['arm']) for s in p['rows'] for e in p['models'] if (e['kind'],e['seed'])==(s['kind'],s['seed'])};assert len(keys)==3600 and reference_count==1200
nativekeys={(s['pair_id'],s['endpoint']) for s in p['rows']};candidateinputs={(s['kind'],s['seed'],s['pair_id'],s['endpoint']) for s in p['rows']};assert r['unique_native_inputs']==len(nativekeys);assert r['forward_count']==len(nativekeys)+5*len(candidateinputs)+6*sum(not x[0] for x in directions.values())
for s in r['summaries']:
 rows=[x for x in r['records'] if all(x[k]==s[k] for k in ['kind','seed','arm','context']) and (s['family'] is None or x['family']==s['family'])];assert len(rows)==s['endpoints'];assert sum(not x['noop'] for x in rows)==s['nonzero'];close(sum(x['R'] for x in rows)/len(rows),s['R']);close(sum(x['intact_KL'] for x in rows)/len(rows),s['intact_KL'])
 for key,field in [('native_response_energy_sum','native_response_energy'),('margin_error_sum','margin_squared_error'),('native_margin_energy_sum','native_margin_energy')]:close(sum(x[field] for x in rows),s[key])
 for field in ['native_binary','native_top1']:close(sum(x[field] for x in rows)/len(rows),s[field+'_rate'])
 for key,num,den in [('relative_response_error',sum(x['R'] for x in rows),s['native_response_energy_sum']),('relative_margin_error',s['margin_error_sum'],s['native_margin_energy_sum'])]:
  if den:close(s[key],num/den)
  else:assert s[key] is None
out=dict(status='passed',result_sha256=sha(rp),paired_context_directions=len(protocolkeys),records=len(keys),reference_rows_replayed=reference_count,samples=len(samples),summaries=len(r['summaries']),noops_per_context=sum(x[2] for x in fixed.values()),max_metric_error=max(errors),scope='Independent exact-token pairing, sampled fixed-task-delta mapping to independently audited M28 deltas, NumPy metrics/competence, every hash/energy/noop/reference invariant and full inventory/aggregate/forward-count accounting. No independent transformer implementation.');(P/'wrapper_crossover_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
