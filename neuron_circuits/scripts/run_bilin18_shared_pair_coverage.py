"""H689: all-pair support coverage with higher combinations held out."""
import json,time,hashlib,itertools
from pathlib import Path
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from feature_structures.models import SharedGroups
from run_feature_structure_synthetic import SeparateBlocks
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_shared_pair_coverage'
def main():
 torch.set_num_threads(6);start=time.time();paths={'original':ART/'feature_structure_synthetic_captures.pt','implementation':ROOT/'scripts/feature_structures/models.py','baseline':ROOT/'scripts/run_feature_structure_synthetic.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};old=torch.load(paths['original'],weights_only=False);B,M=old['true_dictionary'],old['true_membership'];trainsets=[s for n in [1,2] for s in itertools.combinations(range(4),n)];testsets=[s for n in [3,4] for s in itertools.combinations(range(4),n)];torch.manual_seed(689)
 def sample(sets,n):
  g=torch.zeros(n,4);which=torch.randint(len(sets),(n,))
  for i,k in enumerate(which):g[i,list(sets[int(k)])]=1
  a=torch.randn(n,8);q=(g@M>0).float();return (a*q)@B.T,g
 tx,tg=sample(trainsets,4096);vx,vg=sample(testsets,2048);cfg=dict(experiment='H689',hashes=hashes,design='Same H687 teacher; all singleton and all pair supports train, all triple/all4 supports heldout.4096train/2048test samples,3000Adam steps,batch128,3seeds. True group gates supplied. No teacher-basis repair during fitting/evaluation. Same3variants retained.',limitations='Known gates, unmatched model parameter/active dimension budgets. Coverage repair tests generalization mechanism; not unsupervised group discovery or LLM superiority. Larger-support test distribution differs from H687.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];states={}
 for seed in [11,12,13]:
  for kind in ['shared_known_membership','shared_learned_membership','separate_blocks']:
   torch.manual_seed(seed);m=SeparateBlocks() if kind=='separate_blocks' else SharedGroups(16,8,4,2,3,initial_membership=M if kind=='shared_known_membership' else None);opt=torch.optim.Adam(m.parameters(),lr=.003);gen=torch.Generator().manual_seed(seed+100)
   for step in range(3000):
    ii=torch.randint(len(tx),(128,),generator=gen);x,g=tx[ii],tg[ii];out=m(x,g) if kind=='separate_blocks' else m(x,gates=g,membership=M if kind=='shared_known_membership' else None)['reconstruction'];loss=F.mse_loss(out,x);opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():
    def evaluate(x,g):return m(x,g) if kind=='separate_blocks' else m(x,gates=g,membership=M if kind=='shared_known_membership' else None)['reconstruction']
    rec=dict(kind=kind,seed=seed,train_relative_mse=float((evaluate(tx,tg)-tx).square().sum()/tx.square().sum()),heldout_relative_mse=float((evaluate(vx,vg)-vx).square().sum()/vx.square().sum()),parameters=sum(p.numel() for p in m.parameters()))
    if kind!='separate_blocks':
     D=m.dictionary_matrix();score=(B.T@D).abs();aa,bb=linear_sum_assignment(-score.numpy());rec['matched_atom_cosine']=float(score[aa,bb].mean());res=m(vx,gates=vg,membership=M if kind=='shared_known_membership' else None);learned=res['membership'][:,bb];rec['membership_entry_accuracy_after_atom_matching']=float((learned==M).float().mean());rec['mean_active_atoms']=float(res['support'].sum(-1).mean())
    else:rec['mean_active_atoms']=float(vg.sum(-1).mean()*3)
    records.append(rec);states[kind,seed]=m.state_dict();print(rec,flush=True)
 co=torch.zeros(8,8,dtype=torch.bool)
 for s in trainsets:q=(M[list(s)].sum(0)>0);co|=q[:,None]&q[None,:]
 assert co.all();audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},all_atom_pairs_covered=True,finite=all(torch.isfinite(torch.tensor(r['heldout_relative_mse'])).item() for r in records),whole_model_forwards=0);torch.save(dict(states=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits)
if __name__=='__main__':main()
