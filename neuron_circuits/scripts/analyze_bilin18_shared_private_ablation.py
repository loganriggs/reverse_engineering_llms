from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_shared_private_ablation';j=json.loads((A/f'{S}.json').read_text());old=json.loads((A/'bilin18_shared_private_response_groups.json').read_text());key=lambda r:(r['task'],r['id'],r['amp'],r['state'],r['arm']);ref={key(r):r for r in j['records'] if r['dictionary']=='full'};prior={key(r):r for r in old['records'] if r['dictionary']=='sharedprivate'};replay=max(abs(r['logprob_response']-prior[key(r)]['logprob_response']) for r in j['records'] if r['dictionary']=='sharedprivate');assert replay<1e-4
rows=[];effects=[]
for task in ['copy','ioi']:
 for group in ['all',0,1]:
  rr=[r for r in j['records'] if r['task']==task and (group=='all' or r['selected_group']==group)];idx={(key(r),r['dictionary']):r for r in rr};kk=sorted({key(r) for r in rr});zeroerr=np.array([idx[k,'zero']['logprob_response']-ref[k]['logprob_response'] for k in kk]);den=np.linalg.norm(zeroerr)
  for name in ['zero','sharedprivate','shared_only','private_only','wrong_private']:
   err=np.array([idx[k,name]['logprob_response']-ref[k]['logprob_response'] for k in kk]);change=np.array([idx[k,name]['logprob_response']-idx[k,'sharedprivate']['logprob_response'] for k in kk]);rows.append({'task':task,'group':group,'representation':name,'variants':len(kk),'lexical_items':len({k[1].split('-')[3] if task=='copy' else k[1].split('-')[2] for k in []}) if False else len({r['item'] for r in rr}),'error_RMS':float(np.sqrt(np.mean(err**2))),'module_relative_error':float(np.linalg.norm(err)/den),'ablation_change_RMS':float(np.sqrt(np.mean(change**2)))})
  interaction=np.array([idx[k,'sharedprivate']['logprob_response']-idx[k,'shared_only']['logprob_response']-idx[k,'private_only']['logprob_response']+idx[k,'zero']['logprob_response'] for k in kk]);effects.append({'task':task,'group':group,'nonadditive_output_RMS':float(np.sqrt(np.mean(interaction**2)))})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':rows,'interactions':effects,'frozen_H815_replay_max_abs':replay,'limitations':'Gates frozen before deletion; group cells are unequal correlated variants, sometimes few lexical items. Module-relative errors compare to uncompressed response, not uniquely attributable causal shares. Wrong-private swaps preserve dimension but not norm.'},indent=2)+'\n');print('frozen replay',replay)
for r in rows:print(r)
print(effects)
