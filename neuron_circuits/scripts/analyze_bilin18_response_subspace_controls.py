from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_response_subspace_controls';j=json.loads((A/f'{S}.json').read_text());rows=[];gaps=[]
key=lambda r:(r['task'],r['id'],r['amp'],r['state'],r['arm'])
ref={key(r):r for r in j['records'] if r['dictionary']=='full'}
for task in ['copy','ioi']:
 for name,rank in [('full',1152),('zero',0)]+[(n,r) for n in ['shared','specific','cross_task','random8130','random8131','random8132'] for r in [8]]:
  rr=[r for r in j['records'] if r['task']==task and r['dictionary']==name and r['rank']==rank];gg=[r for r in j['geometry'] if r['task']==task and r['dictionary']==name and r['rank']==rank];err=np.array([r['logprob_response']-ref[key(r)]['logprob_response'] for r in rr]);truth=np.array([ref[key(r)]['logprob_response'] for r in rr]);rows.append({'task':task,'dictionary':name,'rank':rank,'mean_relative_input_error':float(np.mean([g['relative_input_error'] for g in gg])),'logprob_error_RMS':float(np.sqrt(np.mean(err**2))),'logprob_error_relative_RMS':float(np.linalg.norm(err)/max(np.linalg.norm(truth),1e-30)),'accuracy':float(np.mean([r['correct'] for r in rr]))})
  for amp in sorted({r['amp'] for r in rr}):
   x=[r for r in rr if r['amp']==amp];v={arm:np.mean([r['logprob_response'] for r in x if r['arm']==arm and r['context']])-np.mean([r['logprob_response'] for r in x if r['arm']==arm and not r['context']]) for arm in ['own','input_swap']};gaps.append({'task':task,'dictionary':name,'rank':rank,'amp':amp,'own_gap':float(v['own']),'input_swap_gap':float(v['input_swap']),'swap_contrast':float(v['input_swap']-v['own'])})
for row in rows:
 zero=next(x['logprob_error_RMS'] for x in rows if x['task']==row['task'] and x['dictionary']=='zero');row['error_relative_to_tested_module_contribution_RMS']=row['logprob_error_RMS']/zero
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':rows,'gaps':gaps,'limitations':'Only three held-out lexical items per task; descriptive. Pool two doses and response arms in RMS error. Specific rank r uses twice total decoder columns of shared rank r, both activate r. Shared2r/specificr matches total columns but differs active dimension.'},indent=2)+'\n');print(j['config']['split']);print(j['audits'])
for r in rows:print(r)
for r in gaps:
 if r['dictionary']=='full' or r['rank'] in [8,16]:print(r)
