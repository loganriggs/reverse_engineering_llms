"""H690: structural rank bounds and fixed/live membership continuation."""
import json,time,hashlib,itertools
from pathlib import Path
import torch
from feature_structures.models import SharedGroups,binary_gate
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_shared_membership_failure'
def main():
 torch.set_num_threads(6);start=time.time()
 paths={'teacher':ART/'feature_structure_synthetic_captures.pt','fits':ART/'bilin18_shared_pair_coverage_captures.pt','implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)}
 hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
 teacher=torch.load(paths['teacher'],weights_only=False);B,M=teacher['true_dictionary'],teacher['true_membership'];states=torch.load(paths['fits'],weights_only=False)['states']
 trainsets=[s for n in [1,2] for s in itertools.combinations(range(4),n)];testsets=[s for n in [3,4] for s in itertools.combinations(range(4),n)]
 torch.manual_seed(689)
 def sample(sets,n):
  g=torch.zeros(n,4);which=torch.randint(len(sets),(n,))
  for i,k in enumerate(which):g[i,list(sets[int(k)])]=1
  a=torch.randn(n,8);return (a*(g@M>0))@B.T,g
 tx,tg=sample(trainsets,4096);vx,vg=sample(testsets,2048)
 cfg=dict(experiment='H690',hashes=hashes,steps=6000,lr=.001,batch=128,design='Same H689 data, supplied gates. Continue weights with fresh Adam, either frozen learned membership or live membership. Exact Gaussian expected errors and rank lower bounds per support. No heldout selection.',limitations='Continuation not exact optimizer resume; learned structure and known gates, not unsupervised discovery. Rank bounds may be loose.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];saved={}
 def evaluate(m):
  with torch.no_grad():
   L=binary_gate(m.membership_logits,3);D=m.dictionary_matrix();E=m.coordinate_encoder.weight;cells=[]
   for split,sets in [('train',trainsets),('heldout',testsets)]:
    for s in sets:
     q=(L[list(s)].sum(0)>0).float();trueq=(M[list(s)].sum(0)>0).float();T=B*trueq;mapping=(D*q)@E;bias=(D*q)@m.coordinate_encoder.bias+m.bias
     error=((mapping@T-T).square().sum()+bias.square().sum())/trueq.sum()
     cells.append(dict(split=split,groups=s,true_dimension=int(trueq.sum()),available_atoms=int(q.sum()),rank_lower_bound=float(torch.clamp(trueq.sum()-q.sum(),min=0)/trueq.sum()),expected_relative_mse=float(error)))
   return dict(membership=L.tolist(),cells=cells,train_relative_mse=float((m(tx,gates=tg)['reconstruction']-tx).square().sum()/tx.square().sum()),heldout_relative_mse=float((m(vx,gates=vg)['reconstruction']-vx).square().sum()/vx.square().sum()))
 for seed in [11,12,13]:
  for mode in ['fixed','live']:
   m=SharedGroups(16,8,4,2,3);m.load_state_dict(states['shared_learned_membership',seed]);initial=evaluate(m);fixed=binary_gate(m.membership_logits,3).detach();opt=torch.optim.Adam(m.parameters(),lr=.001);gen=torch.Generator().manual_seed(seed+69000)
   for step in range(6000):
    ii=torch.randint(len(tx),(128,),generator=gen);out=m(tx[ii],gates=tg[ii],membership=fixed if mode=='fixed' else None);loss=(out['reconstruction']-tx[ii]).square().mean();opt.zero_grad();loss.backward();opt.step()
   final=evaluate(m);rec=dict(seed=seed,mode=mode,initial=initial,final=final);records.append(rec);saved[seed,mode]=m.state_dict();print(dict(seed=seed,mode=mode,initial_error=initial['heldout_relative_mse'],final_error=final['heldout_relative_mse'],membership_changed=initial['membership']!=final['membership']),flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['final']['heldout_relative_mse'])) for r in records),whole_model_forwards=0)
 audits['finite']=bool(audits['finite']);assert all(audits['hashes_unchanged'].values()) and audits['finite']
 torch.save(dict(states=saved),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
