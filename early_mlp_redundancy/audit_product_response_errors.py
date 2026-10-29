"""Independent torch log-softmax/linear-contrast Gram audit of M94."""
import json,sys,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from fit_pca_encoder import sha,V
P=V.parent
FIELDS=['A2','B2','I2','E2','AB2','AI2','BI2','additive2']
def main():
 torch.set_num_threads(1);cohort=sys.argv[1];assert cohort in ['whole','source'];stem='product_subset_factorial' if cohort=='whole' else 'product_subset_source_factorial';ns='whole_task_native' if cohort=='whole' else 'source_holdout_native';rp=V/f'product_error_interaction_{cohort}.json';d=json.loads(rp.read_text());pp=V/'product_subset_error_interaction_protocol.json';p=json.loads(pp.read_text());assert d['protocol_sha256']==sha(pp) and d['source_sha256']==sha(P/'decompose_product_response_errors.py') and d['input_sha256']==sha(V/(stem+'.json'))
 for name,digest in p['source_hashes'].items():assert sha(V/name)==digest
 src=json.loads((V/(stem+'.json')).read_text());native=json.loads((V/(ns+'.json')).read_text());assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True);targets={x['sample_key']:x for x in native['records']};models=json.loads((V/'product_subset_models.json').read_text())['models'];rows={(r['kind'],r['seed'],r['arm'],r['key'],r['region']):r for r in d['rows']};assert len(rows)==len(d['rows']);assert d['ineligible']==src['ineligible'];checked=set();maxdiff=0.;maxvec=0.
 T=torch.tensor([[-1,1,0,0],[-1,0,1,0],[1,-1,-1,1],[-1,0,0,1],[-2,1,1,0]],dtype=torch.float64)
 def check(a,b):
  nonlocal maxdiff
  maxdiff=max(maxdiff,abs(a-b));assert np.isclose(a,b,rtol=1e-7,atol=1e-10),(a,b)
 def logp(t):return torch.log_softmax(t.double(),dim=-1)
 independent={}
 for file in src['files']:
  assert sha(file['path'])==file['sha256'];b=torch.load(file['path'],weights_only=True,mmap=True);pid=b['pair_id']
  for ep in [0,1]:
   endpoint=pid+f'|{ep}';nl=logp(saved['baselines'][endpoint]['logits']);prob=nl.exp()
   for e in models:
    mk=tuple(e[k] for k in ['kind','seed','arm']);ms=':'.join(map(str,mk));bl=logp(b['baselines'][ms+f'|{ep}'])
    for region in ['changed_input_positions','final_position','identity']:
     target=targets[endpoint+'|1|'+region];key=mk+(endpoint,region)
     if target['status']!='evaluated':assert key not in rows;continue
     assert key not in checked;checked.add(key)
     R=torch.stack([logp(saved['changed'][target['sample_key']])-nl,logp(b['changed'][ms+f'|{ep}|{region}|reconstructed_native'])-bl,logp(b['changed'][ms+f'|{ep}|{region}|native_reconstructed'])-nl,logp(b['changed'][ms+f'|{ep}|{region}|reconstructed_reconstructed'])-bl])
     C=T@R;G=(C*prob)@C.T;vi=float((C[3]-C[0]-C[1]-C[2]).abs().max());maxvec=max(maxvec,vi);assert vi<=1e-10
     vals={'A2':float(G[0,0]),'B2':float(G[1,1]),'I2':float(G[2,2]),'E2':float(G[3,3]),'AB2':float(2*G[0,1]),'AI2':float(2*G[0,2]),'BI2':float(2*G[1,2]),'additive2':float(G[4,4]),'teacher2':float((prob*R[0]**2).sum())}
     for field,value in vals.items():check(value,rows[key][field])
     check(vals['E2'],math.fsum(vals[k] for k in ['A2','B2','I2','AB2','AI2','BI2']));independent[key]=vals
  print('AUDIT_INTERACTION',cohort,len(checked),flush=True)
 assert checked==set(rows)
 qualified={x['family'] for x in json.loads((V/'whole_task_native_comparison.json').read_text())['summaries'] if x['competent']} if cohort=='whole' else set();groups=defaultdict(list)
 for key,x in rows.items():
  scopes=['all','family:'+x['family'],'format:'+x['family']+':'+x['format']]+(['contrast_qualified'] if x['family'] in qualified else [])
  for scope in scopes:groups[key[:3]+(scope,x['region'])].append((x,independent[key]))
 seen=set()
 for s in d['summaries']:
  key=tuple(s[k] for k in ['kind','seed','arm','scope','region']);assert key not in seen;seen.add(key);rs=groups[key];assert len(rs)==s['n'] and sum(x[1]['teacher2']==0 for x in rs)==s['null_targets'];den=math.fsum(x[1]['teacher2'] for x in rs);check(den,s['teacher2']);fg=defaultdict(list)
  for x,val in rs:fg[x['family']].append(val)
  for f in FIELDS:
   total=math.fsum(x[1][f] for x in rs);check(total,s[f+'_sum']);ratios=[]
   for fs in fg.values():
    fd=math.fsum(x['teacher2'] for x in fs)
    if fd:ratios.append(math.fsum(x[f] for x in fs)/fd)
   if den:check(total/den,s[f+'_pooled'])
   else:assert s[f+'_pooled'] is None
   if ratios:check(math.fsum(ratios)/len(ratios),s[f+'_family'])
   else:assert s[f+'_family'] is None
 assert seen==set(groups)
 out={'status':'passed','cohort':cohort,'source_sha256':sha(__file__),'result_sha256':sha(rp),'protocol_sha256':sha(pp),'conditions_checked':len(checked),'summaries_checked':len(seen),'max_scalar_discrepancy':maxdiff,'max_vector_identity_absolute_error':maxvec,'scope':'Independent torch log-softmax, fixed linear contrasts and weighted Gram entries from every saved response; independent fsum aggregation. Not an independent transformer implementation. No new forwards/fitting.'}
 dest=V/f'product_error_interaction_{cohort}_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',cohort,len(checked),maxdiff,flush=True)
if __name__=='__main__':main()
