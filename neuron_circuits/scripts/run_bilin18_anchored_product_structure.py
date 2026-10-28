"""H948 discovery-only product selection and gauge-aware cross-seed matching."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha
from pathlib import Path
import torch,numpy as np,json,time
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
S='bilin18_anchored_product_structure';KS=[1,2,4,8,16,32,64]
def factors(state):return F.normalize(state['left'].double(),dim=0),F.normalize(state['right'].double(),dim=0),state['out'].double()
def codes(x,d,L,R):
 u=F.rms_norm(x.double(),(1152,),eps=torch.finfo(torch.float32).eps);v=F.rms_norm((x+d).double(),(1152,),eps=torch.finfo(torch.float32).eps);return (v@L)*(v@R)-(u@L)*(u@R)
def tensor_cos(a,b):
 L,R,O=a;l,r,o=b;inner=.5*((L.T@l)*(R.T@r)+(L.T@r)*(R.T@l))*(O@o.T);na=((1+(L*R).sum(0).square())*.5*O.square().sum(1)).sqrt();nb=((1+(l*r).sum(0).square())*.5*o.square().sum(1)).sqrt();return inner/(na[:,None]*nb[None,:]).clamp_min(1e-30)
def greedy(G):
 mask=torch.ones(len(G),dtype=torch.double);order=[]
 for _ in range(len(G)):
  benefit=2*G@mask-G.diag();benefit[mask==0]=-float('inf');j=int(benefit.argmax());order.append(j);mask[j]=0
 return order
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);paths={'runner':Path(__file__),'fits':ART/'bilin18_supervised_relay_anchors_fits.pt','anchors':ART/'bilin18_supervised_relay_anchors_anchors.pt','evaluation':ART/'bilin18_distilled_relay_mechanisms_captures.pt'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H948','hashes':hashes,'selection':'For each model/task, greedily approximate its OWN complete discovery-anchor finite update with subsets of whole bilinear products. No coefficients refit; order frozen before evaluation. All7 counts retained. Not selection on teacher/evaluation effects.','matching':'Cosine of full output⊗symmetric-input bilinear tensor; analytical Gram handles left/right swaps and compensating signs, independent of factor norm gauges. Hungarian permutation from tensors only.16 signed ambient coordinate permutation nulls of second model, same transform input/output.','limits':'Supervised student and task-conditioned subset selection; model weights highdimensional, product count not total description length. Discovery-preserving subsets need not preserve teacher computation or downstream output.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');fits=torch.load(paths['fits'],weights_only=False);anc=torch.load(paths['anchors'],weights_only=False)['anchors'];ev=torch.load(paths['evaluation'],weights_only=False);orders={};records=[];diag=[];replay=0.;relative_replay=0.;allfactor={};subsets={}
 for (arm,seed),state in fits['models'].items():
  L,R,O=factors(state);allfactor[arm,seed]=(L,R,O);evalcode=codes(ev['inputs'],ev['deltas'],L,R);full=evalcode@O;replay=max(replay,float((full-fits['predicted_effects'][arm,seed].double()).abs().max()));relative_replay=max(relative_replay,float((full-fits['predicted_effects'][arm,seed].double()).norm()/fits['predicted_effects'][arm,seed].double().norm()));Cout=O@O.T
  for fam,a in anc.items():
   dc=codes(a['x'],a['delta'],L,R);G=(dc.T@dc)*Cout;order=greedy(G);orders[arm,seed,fam]=order;den=float(G.sum());diag.append(dict(arm=arm,seed=seed,family=fam,discovery_sum_individual_energy_over_sum_energy=float(G.diag().sum()/G.sum())))
   ix=[n for n,j in enumerate(ev['jobs']) if j['family']==fam]
   for k in KS:
    ids=order[:k];p=evalcode[:,ids]@O[ids];subsets[arm,seed,fam,k]=p.float();delta=dc[:,ids]@O[ids]-dc@O;discovery_error=float(delta.norm()/(dc@O).norm())
    for kind in ['full','rank1']:
     ii=[n for n in ix if ev['jobs'][n]['arm']==kind];truth=ev['true_effects'][ii].double();records.append(dict(arm=arm,seed=seed,family=fam,intervention=kind,k=k,discovery_student_error=discovery_error,evaluation_student_error=float((p[ii]-full[ii]).norm()/full[ii].norm()),evaluation_teacher_error=float((p[ii]-truth).norm()/truth.norm())))
 matches=[];nulls=[];symerr=0.;gen=torch.Generator().manual_seed(948)
 for arm in ['continued','anchor_endpoints','anchor_effects']:
  a=allfactor[arm,0];b=allfactor[arm,1];C=tensor_cos(a,b);ri,ci=linear_sum_assignment(-C.numpy());mapping=dict(zip(ri.tolist(),ci.tolist()));symerr=max(symerr,float((tensor_cos(a,a).diag()-1).abs().max()),float((tensor_cos((a[1],a[0],a[2]),b)-C).abs().max()),float((tensor_cos((-a[0],a[1],-a[2]),b)-C).abs().max()))
  for fam in anc:
   for k in [4,8,16]:
    sa=orders[arm,0,fam][:k];sb=set(orders[arm,1,fam][:k]);mapped={mapping[i] for i in sa};matches.append(dict(arm=arm,family=fam,k=k,selected_tensor_cosine=float(C[sa,[mapping[i] for i in sa]].mean()),selected_support_overlap=len(mapped&sb)/k,all_matched_tensor_mean=float(C[ri,ci].mean())))
  for rep in range(16):
   perm=torch.randperm(1152,generator=gen);sign=torch.randint(0,2,(1152,),generator=gen)*2-1;l,r,o=b;rot=(l[perm]*sign[:,None],r[perm]*sign[:,None],o[:,perm]*sign[None]);N=tensor_cos(a,rot);ii,jj=linear_sum_assignment(-N.numpy());nulls.append(dict(arm=arm,rep=rep,mean_tensor_cosine=float(N[ii,jj].mean())))
 print('AUDIT',symerr,replay,relative_replay,flush=True);assert symerr<1e-10 and relative_replay<1e-4
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'tensor_gauge_checks_max_abs':symerr,'double_product_sum_vs_float_student_max_abs':replay,'double_product_sum_vs_float_student_relative_norm':relative_replay,'finite':all(np.isfinite(r['evaluation_teacher_error']) for r in records),'selection_uses_evaluation':False,'refitting':False,'whole_model_forwards':0,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'];torch.save(dict(orders=orders,predicted_subset_effects=subsets),ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,diagnostics=diag,matches=matches,nulls=nulls,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(matches,audits,flush=True)
if __name__=='__main__':main()
