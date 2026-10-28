from pathlib import Path
import torch,json
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_unsupervised_response_groups';j=json.loads((A/f'{S}.json').read_text());torch.set_num_threads(6);train=[];test=[]
for task,stem,amps in [('copy','bilin18_copy_small_dose_response',[.05,.1]),('ioi','bilin18_indirect_object_response_cross',[.25,.5])]:
 c=torch.load(A/f'{stem}_captures.pt',weights_only=False)
 for key,v in c['captures'].items():
  e=c['examples'][key[0]]
  if e['format']!=0 or key[-1] not in amps:continue
  dx=v['premlp'].double()-c['captures'][(*key[:-1],0.)]['premlp'].double();(test if e['item'] in j['config']['split'][task]['heldout'] else train).append(dx/dx.norm())
X=torch.stack(train);Y=torch.stack(test);fits=[]
for seed in range(8140,8145):
 labels=torch.randint(2,(len(X),),generator=torch.Generator().manual_seed(seed))
 for it in range(50):
  bs=[torch.linalg.svd(X[labels==g],full_matrices=False).Vh[:4].T.contiguous() for g in [0,1]];energy=torch.stack([(X@b).square().sum(1) for b in bs],1);new=energy.argmax(1)
  if torch.equal(new,labels):break
  labels=new
 pred=torch.stack([(Y@b).square().sum(1) for b in bs],1).argmax(1);loss=float((1-energy.max(1).values).mean());expected=next(r['loss'] for r in j['groupfit']['restarts'] if r['seed']==seed);assert abs(loss-expected)<1e-12;fits.append({'seed':seed,'loss':loss,'train':labels,'held':pred})
best=next(r for r in fits if r['seed']==j['groupfit']['selected_seed']);out=[]
for f in fits:
 row={'seed':f['seed'],'train_loss':f['loss']}
 for split in ['train','held']:
  disagree=float((f[split]!=best[split]).double().mean());row[split+'_agreement_up_to_global_label_swap']=max(disagree,1-disagree)
 out.append(row)
(A/f'{S}_stability.json').write_text(json.dumps({'restart_comparison':out,'training_examples':len(X),'heldout_examples':len(Y),'note':'Assignment agreement after best global binary label permutation; variants are correlated, not independent samples. Initialization instability is not proof that no meaningful grouping exists.'},indent=2)+'\n')
print(out)
