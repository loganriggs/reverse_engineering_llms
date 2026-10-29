"""M94 fixed all-case vector decomposition from already-saved M93 outputs."""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from fit_pca_encoder import sha,V
from audit_pca_factorial import lp
P=V.parent
FIELDS=['A2','B2','I2','E2','AB2','AI2','BI2','additive2']
def main():
 torch.set_num_threads(1);cohort=sys.argv[1];assert cohort in ['whole','source'];stem='product_subset_factorial' if cohort=='whole' else 'product_subset_source_factorial';ns='whole_task_native' if cohort=='whole' else 'source_holdout_native';pp=V/'product_subset_error_interaction_protocol.json';protocol=json.loads(pp.read_text())
 for f,h in protocol['source_hashes'].items():assert sha(V/f)==h
 execution=json.loads((V/'product_subset_causal_execution.json').read_text())
 for f in [ns+'.json',ns+'_protocol.json']:assert sha(V/f)==execution['input_hashes'][f]
 r=json.loads((V/(stem+'.json')).read_text());audit=json.loads((V/(stem+'_audit.json')).read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(V/(stem+'.json'));models=json.loads((V/'product_subset_models.json').read_text())['models'];native=json.loads((V/(ns+'.json')).read_text());assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True);targets={x['sample_key']:x for x in native['records']};pairs={x['id']:x for x in json.loads((V/(ns+'_protocol.json')).read_text())['pairs']};qualified={x['family'] for x in json.loads((V/'whole_task_native_comparison.json').read_text())['summaries'] if x['competent']} if cohort=='whole' else set()
 old={tuple(x[k] for k in ['kind','seed','arm','key','region','corner']):x for x in r['rows']};rows=[];maxvec=0.;maxenergy=0.
 def check(a,b):
  nonlocal maxenergy
  maxenergy=max(maxenergy,abs(a-b));assert np.isclose(a,b,rtol=1e-7,atol=1e-10),(a,b)
 for file in r['files']:
  assert sha(file['path'])==file['sha256'];b=torch.load(file['path'],weights_only=True,mmap=True);pid=b['pair_id'];pair=pairs[pid]
  for ep in [0,1]:
   endpoint=pid+f'|{ep}';n=lp(saved['baselines'][endpoint]['logits']);prob=np.exp(n)
   for e in models:
    mk=tuple(e[k] for k in ['kind','seed','arm']);ms=':'.join(map(str,mk));predbase=lp(b['baselines'][ms+f'|{ep}'])
    for region in ['changed_input_positions','final_position','identity']:
     target=targets[endpoint+'|1|'+region]
     if target['status']!='evaluated':continue
     R00=lp(saved['changed'][target['sample_key']])-n
     R10=lp(b['changed'][ms+f'|{ep}|{region}|reconstructed_native'])-predbase
     R01=lp(b['changed'][ms+f'|{ep}|{region}|native_reconstructed'])-n
     R11=lp(b['changed'][ms+f'|{ep}|{region}|reconstructed_reconstructed'])-predbase
     A=R10-R00;B=R01-R00;I=R11-R10-R01+R00;E=R11-R00
     ve=float(np.max(np.abs(E-(A+B+I))));maxvec=max(maxvec,ve);assert ve<=1e-10
     dot=lambda x,y:float(np.sum(prob*x*y))
     vals=dict(A2=dot(A,A),B2=dot(B,B),I2=dot(I,I),E2=dot(E,E),AB2=2*dot(A,B),AI2=2*dot(A,I),BI2=2*dot(B,I),additive2=dot(A+B,A+B));teacher=dot(R00,R00)
     check(vals['E2'],sum(vals[k] for k in ['A2','B2','I2','AB2','AI2','BI2']));check(vals['additive2'],vals['A2']+vals['B2']+vals['AB2']);check(teacher,target['response_energy'])
     for field,corner in [('A2','reconstructed_native'),('B2','native_reconstructed'),('E2','reconstructed_reconstructed')]:check(vals[field],old[mk+(endpoint,region,corner)]['response_error'])
     row={k:e[k] for k in ['kind','seed','arm','space','method','correction','size','scalar']};row.update(key=endpoint,pair_id=pid,endpoint=ep,family=pair['family'],format=pair.get('format','historical_source_group'),region=region,teacher2=teacher,**vals);rows.append(row)
     if region=='identity':assert teacher==0 and all(v==0 for v in vals.values())
  print('DECOMPOSED',cohort,len(rows),flush=True)
 grouped=defaultdict(list)
 for x in rows:
  scopes=['all','family:'+x['family'],'format:'+x['family']+':'+x['format']]+(['contrast_qualified'] if x['family'] in qualified else [])
  for scope in scopes:grouped[x['kind'],x['seed'],x['arm'],scope,x['region']].append(x)
 summaries=[]
 for key,rs in grouped.items():
  x=rs[0];s={k:x[k] for k in ['kind','seed','arm','space','method','correction','size','scalar']};s.update(scope=key[3],region=key[4],n=len(rs),null_targets=sum(z['teacher2']==0 for z in rs));den=sum(z['teacher2'] for z in rs);s['teacher2']=den
  familygroups=defaultdict(list)
  for z in rs:familygroups[z['family']].append(z)
  for f in FIELDS:
   total=sum(z[f] for z in rs);ratios=[]
   for fs in familygroups.values():
    fd=sum(z['teacher2'] for z in fs)
    if fd:ratios.append(sum(z[f] for z in fs)/fd)
   s[f+'_sum']=total;s[f+'_pooled']=total/den if den else None;s[f+'_family']=float(np.mean(ratios)) if ratios else None
  summaries.append(s)
 expected=39032 if cohort=='whole' else 9996;assert len(rows)==expected
 out=dict(status='completed_pending_audit',cohort=cohort,source_sha256=sha(__file__),metric_helper_sha256=sha(P/'audit_pca_factorial.py'),protocol_sha256=sha(pp),input_sha256=sha(V/(stem+'.json')),input_audit_sha256=sha(V/(stem+'_audit.json')),rows=rows,summaries=summaries,ineligible=r['ineligible'],max_vector_identity_absolute_error=maxvec,max_energy_replay_absolute_error=maxenergy,scope='All fixed cases. Native-probability-weighted response errors, not native channel contributions. Additive2 is an algebraic diagnostic, not a new intervention. Zero denominators stay null; no fitting/forwards.')
 dest=V/f'product_error_interaction_{cohort}.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('DONE',cohort,len(rows),maxvec,maxenergy,flush=True)
if __name__=='__main__':main()
