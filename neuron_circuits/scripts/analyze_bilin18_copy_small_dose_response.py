from pathlib import Path
import json,numpy as np,torch
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_copy_small_dose_response';rng=np.random.default_rng(811);summary=[];doses=[]
def boot(v):
 v=np.array(v);b=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);return {'mean':float(v.mean()),'CI':np.quantile(b,[.025,.975]).tolist()}
for stem,amps in [(S,[.025,.05,.1]),('bilin18_mlp_input_baseline_fresh',[.25,.5]),('bilin18_indirect_object_response_cross',[.25,.5])]:
 j=json.loads((A/f'{stem}.json').read_text());ioi='indirect_object' in stem;c=torch.load(A/f'{stem}_captures.pt',weights_only=False);ex=c['examples'];caps=c['captures']
 for f in [0,1]:
  for amp in amps:
   ratios=[]
   for k,v in caps.items():
    if k[-1]!=amp or ex[k[0]]['format']!=f:continue
    b=caps[(*k[:-1],0.)]['premlp'].double();ratios.append(float((v['premlp'].double()-b).norm()/b.norm()))
   doses.append({'source':stem,'format':f,'amplitude':amp,'relative_displacement_mean':float(np.mean(ratios)),'min':min(ratios),'max':max(ratios)})
   rr=[r for r in j['records'] if r['format']==f and r['delta']==amp];ids=sorted({r['item'] for r in rr});gaps={};metric='first_logprob_response' if ioi else 'response';condition=lambda r:r['first_name_correct'] if ioi else not r['context_same_key']
   for arm in ['own','input_swap','baseline_swap','both']:
    x=[r for r in rr if r['arm']==arm];gaps[arm]=np.array([np.mean([r[metric] for r in x if r['item']==i and condition(r)])-np.mean([r[metric] for r in x if r['item']==i and not condition(r)]) for i in ids]);summary.append({'source':stem,'format':f,'amplitude':amp,'arm':arm,'gap':boot(gaps[arm]),'change_from_own':boot(gaps[arm]-gaps['own']),'accuracy':float(np.mean([r['correct'] for r in x]))})
   summary.append({'source':stem,'format':f,'amplitude':amp,'contrast':'input_minus_baseline','value':boot(gaps['input_swap']-gaps['baseline_swap'])})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'doses':doses,'limitations':'Positive dose ladder; realized downstream displacement descriptive, not a randomized dose match. Context contrasts differ by task. Copying f1 below frozen item gate. Historical high-dose copying float32 differences vs current float64; endpoint/logit audits bound numerical error.'},indent=2)+'\n')
for r in doses:
 if r['format']==0:print(r)
for r in summary:
 if r['format']==0 and r['source']==S:print(r)
