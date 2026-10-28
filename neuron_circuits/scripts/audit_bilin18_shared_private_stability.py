from pathlib import Path
import torch,json
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_shared_private_response_groups';j=json.loads((A/f'{S}.json').read_text());v=torch.load(A/f'{S}_bases.pt',weights_only=False);held=[]
for task,stem,amps in [('copy','bilin18_copy_small_dose_response',[.05,.1]),('ioi','bilin18_indirect_object_response_cross',[.25,.5])]:
 c=torch.load(A/f'{stem}_captures.pt',weights_only=False)
 for k,a in c['captures'].items():
  e=c['examples'][k[0]]
  if e['format']!=0 or e['item'] not in j['config']['split'][task]['heldout'] or k[-1] not in amps:continue
  d=a['premlp'].double()-c['captures'][(*k[:-1],0.)]['premlp'].double();held.append(d/d.norm())
X=torch.stack(held);results={}
for tag,fits in v['allfits'].items():
 pred={f['seed']:torch.stack([(X@B).square().sum(1) for B in f['bases']],1).argmax(1) for f in fits};best=pred[j['groupfit'][tag]['selected_seed']];results[tag]=[]
 for f in fits:
  d=float((pred[f['seed']]!=best).double().mean());results[tag].append({'seed':f['seed'],'training_loss':f['loss'],'heldout_assignment_agreement_up_to_label_swap':max(d,1-d)})
(A/f'{S}_stability.json').write_text(json.dumps({'results':results,'n_response_variants':len(X),'note':'Correlated variants; shared component same across groups, so private energy determines shared/private gate.'},indent=2)+'\n');print(results)
