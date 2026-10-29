"""M112 independent reconstruction, candidate delta, response metrics and coverage audit."""
import json,sys
import numpy as np
import torch
from fit_pca_encoder import V,sha


def lp(z):
 z=np.asarray(z,dtype=float);z=z-z.max();return z-np.log(np.exp(z).sum())


def margin(z,g,v):
 def lse(x):m=x.max();return m+np.log(np.exp(x-m).sum())
 return lse(z[g[v]])-lse(z[g[1-v]])


def main():
 torch.set_num_threads(1);cohort=sys.argv[1];rp=V/f'continuous_tasks_{cohort}.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(V.parent/'probe_continuous_tasks.py');assert r['execution_sha256']==sha(V/'continuous_tasks_execution.json')
 stem={'train':'training_causal_coverage','source':'source_holdout_native','whole':'whole_task_native'}[cohort];nr=json.loads((V/(stem+'.json')).read_text());assert sha(nr['samples_path'])==nr['samples_sha256'];native=torch.load(nr['samples_path'],weights_only=True);pairs={p['id']:p for p in json.loads((V/(stem+'_protocol.json')).read_text())['pairs']}
 mr=json.loads((V/'continuous_training_results.json').read_text());states={}
 for entry in mr['models']:
  assert sha(entry['path'])==entry['sha256'];states[entry['seed'],entry['mode']]=torch.load(entry['path'],weights_only=True)
 maxmetric=0.;maxdecode=0.;maxdelta=0.;checked=0;decodes=0;seen=set();intactseen=set()
 def compare(actual,wanted):
  nonlocal maxmetric,checked
  for k,val in wanted.items():
   err=abs(actual[k]-val);maxmetric=max(maxmetric,err)
   if k in ['local_error','local_energy','native_delta_energy','candidate_delta_energy']:assert err/max(1,abs(val))<1e-10,(k,err,val)
   else:assert err<1e-8,(k,err)
   checked+=1
 for file in r['files']:
  assert sha(file['path'])==file['sha256'];b=torch.load(file['path'],weights_only=True);pid=file['pair_id'];pair=pairs[pid]
  for ep in [0,1]:assert torch.equal(b['baselines'][ep],native['baselines'][pid+f'|{ep}']['logits'])
  for key,rec in b['reconstructions'].items():
   seed,mode,ep=key.split('|');seed=int(seed);ep=int(ep);state=states[seed,mode];s={k:t.double().numpy() for k,t in state['state'].items()};mu=state['mean'].double().numpy();scale=state['scale'].double().numpy();w=b['writes'][ep].double().numpy();raw=((w-mu)/scale)@s['encoder'].T+s['encoder_bias'];a=raw.reshape(*raw.shape[:-1],32,4);norm=np.linalg.norm(a,axis=-1)
   if mode=='hard':
    idx=np.argsort(norm,axis=-1)[...,-8:];q=np.zeros_like(norm);np.put_along_axis(q,idx,1,-1);code=(a*q[...,None]).reshape(raw.shape)
   else:
    tau=np.sort(norm,axis=-1)[...,-9,None];code=(a*(np.maximum(norm-tau,0)/np.where(norm>0,norm,1))[...,None]).reshape(raw.shape)
   D=s['dictionary']/np.linalg.norm(s['dictionary'],axis=0);reconstruction=code@D.T*scale+s['bias']*scale+mu;actual=rec['write'].double().numpy();err=np.linalg.norm(actual-reconstruction)/max(np.linalg.norm(actual),1);maxdecode=max(maxdecode,float(err));assert err<1e-5;decodes+=1
   bas=b['baselines'][ep].double().numpy();out=b['intact'][key].double().numpy();nlp=lp(bas);olp=lp(out);g=pair['answer_groups'][ep];variant=pair['variants'][ep];ir=next(x for x in r['intact'] if x['pair_id']==pid and x['seed']==seed and x['mode']==mode and x['endpoint']==ep);intactseen.add((pid,seed,mode,ep))
   compare(ir,dict(KL=float((np.exp(nlp)*(nlp-olp)).sum()),top1_agreement=int(bas.argmax()==out.argmax()),margin_error=margin(out,g,variant)-margin(bas,g,variant),local_error=float(np.square(actual-w).sum()),local_energy=float(np.square(w).sum()),active_mean=float(rec['support'].double().sum(-1).mean())))
  targets=[x for x in nr['records'] if x['pair_id']==pid and x['layer']==1 and x['status']=='evaluated']
  expected=set()
  for target in targets:
   ep=target['endpoint'];region=target['region'];pos=target['recipient_positions'];dp=target['donor_positions'];baseline=b['baselines'][ep].double().numpy();nlp=lp(baseline);prob=np.exp(nlp);teacher=native['changed'][target['sample_key']].double().numpy();teacherresponse=lp(teacher)-nlp;nd=native['deltas'][target['sample_key']]['delta'].double().numpy();g=pair['answer_groups'][ep];variant=pair['variants'][ep];tm=margin(teacher,g,variant)-margin(baseline,g,variant)
   for seed in [71,72]:
    ds={mode:(b['reconstructions'][f'{seed}|{mode}|{1-ep}']['write'][:,dp]-b['reconstructions'][f'{seed}|{mode}|{ep}']['write'][:,pos]).numpy() for mode in ['hard','continuous']};hn=np.linalg.norm(ds['hard'].astype(float));sn=np.linalg.norm(ds['continuous'].astype(float))
    if hn>0:ds['matched_hard']=ds['hard']*np.float32(sn/hn)
    elif sn==0:ds['matched_hard']=ds['hard']
    else:assert any(x.get('seed')==seed and x['pair_id']==pid and x['endpoint']==ep and x['region']==region and x.get('condition')=='matched_hard' for x in r['ineligible'])
    for condition,delta in ds.items():
     key=f'{seed}|{ep}|{region}|{condition}';expected.add(key);saved_delta=b['deltas'][key].double().numpy();err=float(np.linalg.norm(saved_delta-delta)/max(np.linalg.norm(delta),1));maxdelta=max(maxdelta,err);assert err<1e-6
     if condition=='matched_hard':assert abs(np.linalg.norm(saved_delta)-sn)<=1e-6*max(sn,1)
     out=b['changed'][key].double().numpy();response=lp(out)-nlp;cm=margin(out,g,variant)-margin(baseline,g,variant);row=next(x for x in r['rows'] if x['pair_id']==pid and x['seed']==seed and x['endpoint']==ep and x['region']==region and x['condition']==condition);seen.add((pid,seed,ep,region,condition))
     compare(row,dict(response_error=float((prob*np.square(response-teacherresponse)).sum()),teacher_response_energy=float((prob*np.square(teacherresponse)).sum()),margin_error=(cm-tm)**2,teacher_margin_energy=tm**2,margin_change=cm,teacher_margin_change=tm,local_error=float(np.square(saved_delta-nd).sum()),native_delta_energy=float(np.square(nd).sum()),candidate_delta_energy=float(np.square(saved_delta).sum())))
  assert set(b['changed'])==set(b['deltas'])==expected
 assert len(seen)==len(r['rows']) and len(intactseen)==len(r['intact'])
 wanted=json.loads((V/'block_change_roles_protocol.json').read_text())[{'train':'training_pair_ids','source':'source_group_pair_ids','whole':'whole_task_pair_ids'}[cohort]];assert [x['pair_id'] for x in r['files']]==wanted
 assert len(intactseen)==len(wanted)*8
 out=dict(status='passed',cohort=cohort,source_sha256=sha(__file__),results_sha256=sha(rp),metric_checks=checked,decoded_endpoints=decodes,response_vectors=len(seen),complete_saved_key_coverage=True,max_metric_error=maxmetric,max_decode_error=maxdecode,max_delta_error=maxdelta)
 dest=V/f'continuous_tasks_{cohort}_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
