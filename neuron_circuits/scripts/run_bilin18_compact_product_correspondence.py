"""H951: frozen discovery/tensor matching of H949 compact causal products."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha
from run_bilin18_anchored_product_structure import factors,codes,tensor_cos
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import torch,json,numpy as np
S='bilin18_compact_product_correspondence'
def cosine(c,C,o,O):
 inner=(c.T@C)*(o@O.T);a=(c.square().sum(0)*o.square().sum(1)).sqrt();b=(C.square().sum(0)*O.square().sum(1)).sqrt()
 return inner/(a[:,None]*b[None,:]).clamp_min(1e-30)
def sim(a,b):return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
@torch.inference_mode()
def main():
 torch.set_num_threads(6)
 paths={'runner':Path(__file__),'math':ROOT/'scripts/run_bilin18_anchored_product_structure.py','fits':ART/'bilin18_product_complexity_training_fits.pt','anchors':ART/'bilin18_supervised_relay_anchors_anchors.pt','evaluation':ART/'bilin18_distilled_relay_mechanisms_captures.pt'}
 hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H951','hashes':hashes,'design':'All four H949 penalties, frozen two-seed fits and task-specific discovery greedy orders. Hungarian assignment using either full symmetric product tensors or product response matrices on discovery anchors, normalized equally by task teacher response energy. Frozen assignments evaluated on full and rank1 heldout contrasts. Report selected-product agreement, amplitude disagreement and aggregate agreement; do not equate cosine with faithful amplitude.','limits':'Only two seeds, known supervised tasks and repeatedly used development set. Near-zero collapsed fits are negative controls. Shared selected support does not itself establish cross-task computational reuse.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 v=torch.load(paths['fits'],weights_only=False);anc=torch.load(paths['anchors'],weights_only=False)['anchors'];ev=torch.load(paths['evaluation'],weights_only=False)
 rows=[];dominant=[];reuse=[];assignments={};audit=0.;explicit_audit=0.
 for lam in [0.,.01,.1,1.]:
  fs=[factors(v['models'][lam,seed]) for seed in [0,1]];l,r,o=fs[0];L,R,O=fs[1];ac=[];bc=[]
  for f,p in anc.items():
   scale=p['effect'].double().norm();ac.append(codes(p['x'],p['delta'],l,r)/scale);bc.append(codes(p['x'],p['delta'],L,R)/scale)
  a=torch.cat(ac);b=torch.cat(bc);functional=cosine(a,b,o,O);tensor=tensor_cos(fs[0],fs[1]);audit=max(audit,float((cosine(a,a,o,o).diag()-1).abs().max()))
  # Explicit contribution vectors independently check factored functional Gram on a subset.
  aa=(a[:,:3,None]*o[None,:3,:]).permute(1,0,2).reshape(3,-1);bb=(b[:,:4,None]*O[None,:4,:]).permute(1,0,2).reshape(4,-1)
  direct=(aa@bb.T)/(aa.norm(dim=1)[:,None]*bb.norm(dim=1)[None,:]).clamp_min(1e-30)
  explicit_audit=max(explicit_audit,float((direct-functional[:3,:4]).abs().max()))
  for seed in [0,1]:
   orders=v['orders'][lam,seed]
   for k in [1,2,4,8]:reuse.append(dict(lam=lam,seed=seed,k=k,shared_indices=sorted(set(orders['agreement'][:k])&set(orders['sentiment'][:k]))))
  for method,matrix in [('tensor',tensor),('discovery_functional',functional)]:
   ri,ci=linear_sum_assignment(-matrix.numpy());mapping=dict(zip(ri.tolist(),ci.tolist()));assignments[str((lam,method))]=mapping
   for fam in anc:
    sa=v['orders'][lam,0][fam];sb=v['orders'][lam,1][fam]
    for kind in ['full','rank1']:
     ii=[i for i,j in enumerate(ev['jobs']) if j['family']==fam and j['arm']==kind];x=ev['inputs'][ii];d=ev['deltas'][ii];c=codes(x,d,l,r);C=codes(x,d,L,R);test=cosine(c,C,o,O);truth=ev['true_effects'][ii].double()
     for k in [1,2,4,8,16,64]:
      ids=sa[:k];mapped=[mapping[i] for i in ids];other=sb[:k];pa=c[:,ids]@o[ids];pb=C[:,other]@O[other];pm=C[:,mapped]@O[mapped]
      energy=c[:,ids].square().sum(0)*o[ids].square().sum(1);weights=energy/energy.sum().clamp_min(1e-30)
      rows.append(dict(lam=lam,method=method,family=fam,intervention=kind,k=k,selected_support_overlap=len(set(mapped)&set(other))/k,selected_tensor_cosine=float(tensor[ids,mapped].mean()),heldout_matched_product_cosine=float(test[ids,mapped].mean()),heldout_energy_weighted_product_cosine=float((test[ids,mapped]*weights).sum()),independently_selected_sum_cosine=sim(pa,pb),independently_selected_sum_relative_disagreement=float((pa-pb).norm()/pa.norm().clamp_min(1e-30)),mapped_sum_relative_disagreement=float((pa-pm).norm()/pa.norm().clamp_min(1e-30)),seed0_subset_teacher_error=float((pa-truth).norm()/truth.norm()),seed1_subset_teacher_error=float((pb-truth).norm()/truth.norm())))
     # Directly compare top discovery-selected product from each seed, no Hungarian constraint.
     if method=='discovery_functional':
      i,j=sa[0],sb[0];pa=c[:,i,None]*o[i];pb=C[:,j,None]*O[j]
      dominant.append(dict(lam=lam,family=fam,intervention=kind,seed0_index=i,seed1_index=j,tensor_cosine=float(tensor[i,j]),discovery_functional_cosine=float(functional[i,j]),heldout_response_cosine=float(test[i,j]),relative_disagreement=float((pa-pb).norm()/pa.norm().clamp_min(1e-30)),norm_ratio_seed1_over_seed0=float(pb.norm()/pa.norm().clamp_min(1e-30)),seed0_teacher_error=float((pa-truth).norm()/truth.norm()),seed1_teacher_error=float((pb-truth).norm()/truth.norm())))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'functional_self_cosine_max_abs_error':audit,'explicit_functional_cosine_max_abs_error':explicit_audit,'assignments_are_permutations':all(sorted(m.values())==list(range(64)) for m in assignments.values()),'finite':all(np.isfinite(r['heldout_matched_product_cosine']) for r in rows),'whole_model_forwards':0,'test_evaluations':0,'selection_uses_evaluation':False}
 assert audits['hashes_unchanged'] and audits['finite'] and audits['assignments_are_permutations'] and audit<1e-10 and explicit_audit<1e-10
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=rows,dominant=dominant,reuse=reuse,assignments=assignments,audits=audits),indent=2)+'\n')
 print('AUDITS',audits)
 for row in dominant:
  if row['intervention']=='full':print(row)
 for row in rows:
  if row['method']=='discovery_functional' and row['intervention']=='full' and row['k']==8:print('TOP8',row)
 print('REUSE',[r for r in reuse if r['k'] in [1,8]])
if __name__=='__main__':main()
