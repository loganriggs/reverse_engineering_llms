"""Evaluate fixed initial/final direct selectors on all training pairs, no updates."""
from pathlib import Path
p=Path('/home/loganriggs/Coding/neuron_circuits/scripts/run_bilin18_direct_change_supervision.py')
s=p.read_text();exec(compile(s[:s.index("\nfor kind in ['scalar','block','shared']:")],str(p),'exec'),globals())
rows=[]
with torch.no_grad():
 for kind in ['scalar','block','shared']:
  for checkpoint in ['initial','final']:
   selector=Selector(kind)
   if checkpoint=='final':selector.load_state_dict(torch.load(out/f'{kind}.pt',weights_only=False)['state'])
   for p in trainpairs:
    i,d=p['indices'];delta=tuple(states[d][k]-states[i][k] for k in range(3));change,_=direct(selector,delta);yi=suffix(tuple(states[i][k]+change[k] for k in range(3)));yd=suffix(tuple(states[d][k]-change[k] for k in range(3)));value=float((KL(native_train[d].double(),yi.double())+KL(native_train[i].double(),yd.double()))/2);rows.append(dict(kind=kind,checkpoint=checkpoint,pair=p['id'],family=p['family'],mean_bidirectional_KL=value))
summary=[]
for kind in ['scalar','block','shared']:
 for checkpoint in ['initial','final']:
  rr=[r for r in rows if r['kind']==kind and r['checkpoint']==checkpoint];families=sorted({r['family'] for r in rr});means={f:float(np.mean([r['mean_bidirectional_KL'] for r in rr if r['family']==f])) for f in families};summary.append(dict(kind=kind,checkpoint=checkpoint,equal_family_training_KL=float(np.mean(list(means.values()))),family_KL=means))
result=dict(summary=summary,records=rows,fit_rows=0,suffix_sequences=counts['suffix'],scope='All80training pairs, exact hard supports, same loss without small importance penalty. Not validation or discovery.')
(out/'training_fit_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
