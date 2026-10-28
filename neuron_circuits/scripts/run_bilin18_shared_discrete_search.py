"""H691: training-only local discrete search with batched continuous refits."""
import torch,json,time,hashlib,itertools
from pathlib import Path
from feature_structures.models import SharedGroups,binary_gate
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_shared_discrete_search'
def main():
 torch.set_num_threads(2);start=time.time();paths={'teacher':ART/'feature_structure_synthetic_captures.pt','fits':ART/'bilin18_shared_pair_coverage_captures.pt','implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
 teacher=torch.load(paths['teacher'],weights_only=False);B,M=teacher['true_dictionary'],teacher['true_membership'];states=torch.load(paths['fits'],weights_only=False)['states'];torch.manual_seed(689)
 def sample(sets,n):
  g=torch.zeros(n,4);which=torch.randint(len(sets),(n,))
  for i,k in enumerate(which):g[i,list(sets[int(k)])]=1
  a=torch.randn(n,8);return (a*(g@M>0))@B.T,g
 tx,tg=sample([s for n in [1,2] for s in itertools.combinations(range(4),n)],4096);vx,vg=sample([s for n in [3,4] for s in itertools.combinations(range(4),n)],2048)
 cfg=dict(experiment='H691',hashes=hashes,rounds=3,refit_steps=500,lr=.003,batch=128,design='Enumerate all60 single-row membership swaps plus incumbent. Batched independent continuous refits from same incumbent, same minibatches. Accept only best full-training error improvement; retain untouched incumbent as fallback. Supplied true group gates. No heldout selection.',limitations='Local search not global recovery; much greater compute than ST; no matched-budget superiority claim; group gates supplied.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];saved={};max_parity=0.
 def output(x,g,L,D,E,eb,b):
  q=1-(1-g[None,:,:,None]*L[:,None,:,:]).prod(2)
  a=torch.einsum('nd,cad->cna',x,E)+eb[:,None,:]
  return torch.einsum('cna,cda->cnd',a*q,torch.nn.functional.normalize(D,dim=1))+b[:,None,:]
 def score(x,g,L,params):
  with torch.no_grad():
   sums=torch.zeros(len(L))
   for i in range(0,len(x),256):sums+=(output(x[i:i+256],g[i:i+256],L,*params)-x[None,i:i+256]).square().sum((1,2))
   return sums/x.square().sum()
 for seed in [11,12,13]:
  model=SharedGroups(16,8,4,2,3);model.load_state_dict(states['shared_learned_membership',seed]);L=binary_gate(model.membership_logits,3).detach()[None];params=[p.detach()[None].clone() for p in [model.dictionary,model.coordinate_encoder.weight,model.coordinate_encoder.bias,model.bias]]
  with torch.no_grad():
   parity=float((output(tx[:128],tg[:128],L,*params)[0]-model(tx[:128],gates=tg[:128])['reconstruction']).abs().max());max_parity=max(max_parity,parity);assert parity<1e-5
  initial=float(score(tx,tg,L,params)[0]);history=[];gen=torch.Generator().manual_seed(seed+69100)
  for rd in range(3):
   choices=[L[0].clone()];swaps=[None]
   for group in range(4):
    for remove in torch.where(L[0,group]>0)[0].tolist():
     for add in torch.where(L[0,group]==0)[0].tolist():
      ll=L[0].clone();ll[group,remove]=0;ll[group,add]=1;choices.append(ll);swaps.append([group,remove,add])
   choices=torch.stack(choices);assert len(choices)==61 and (choices.sum(-1)==3).all()
   incumbent=float(score(tx,tg,L,params)[0]);ps=[p.repeat(61,*(1 for _ in range(p.ndim-1))).requires_grad_() for p in params];opt=torch.optim.Adam(ps,lr=.003)
   for step in range(500):
    ii=torch.randint(len(tx),(128,),generator=gen);loss=(output(tx[ii],tg[ii],choices,*ps)-tx[None,ii]).square().mean((1,2)).sum();opt.zero_grad();loss.backward();opt.step()
   scores=score(tx,tg,choices,ps);best=int(scores.argmin());accept=float(scores[best])<incumbent
   rec=dict(round=rd,incumbent=incumbent,best_refit=float(scores[best]),selected_swap=swaps[best],accepted=accept,all_training_errors=scores.tolist())
   if accept:L=choices[best:best+1].detach();params=[p[best:best+1].detach().clone() for p in ps]
   history.append(rec);print(dict(seed=seed,**{k:v for k,v in rec.items() if k!='all_training_errors'}),flush=True)
  finaltrain=float(score(tx,tg,L,params)[0]);assert finaltrain<=initial+1e-7
  # Heldout evaluation only after every training-driven choice is complete.
  held=float(score(vx,vg,L,params)[0]);record=dict(seed=seed,initial_train=initial,final_train=finaltrain,final_heldout=held,membership=L[0].tolist(),history=history);records.append(record);saved[seed]=dict(membership=L,parameters=params);print(dict(seed=seed,final_train=finaltrain,final_heldout=held),flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['final_heldout'])).item() for r in records),batched_implementation_max_parity_error=max_parity,training_monotonic=True,whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite']
 torch.save(dict(states=saved),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
