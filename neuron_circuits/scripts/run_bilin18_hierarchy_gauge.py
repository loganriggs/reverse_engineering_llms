"""H697: constructive hierarchy gauge and ancestor-residualized convention."""
import torch,json,time,hashlib,itertools
from pathlib import Path
from run_bilin18_hierarchy_recovery import PARENTS,HierarchyFit
from feature_structures.models import implication_closure
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_hierarchy_gauge'
def ancestors(node):
 result=set(PARENTS[node])
 for p in PARENTS[node]:result.update(ancestors(p))
 return sorted(result)
def residualize(D):
 # D_new=D@A; only ancestor rows enter each node's columns.
 current=D.clone();A=torch.eye(14,dtype=D.dtype)
 for node in range(7):
  aa=[2*p+k for p in ancestors(node) for k in range(2)];cc=[2*node,2*node+1]
  if aa:
   coef=torch.linalg.lstsq(current[:,aa],current[:,cc]).solution;step=torch.eye(14,dtype=D.dtype);step[torch.tensor(aa)[:,None],torch.tensor(cc)[None,:]]-=coef;current=current@step;A=A@step
 return current,A

def main():
 torch.set_num_threads(2);start=time.time();paths={'fits':ART/'bilin18_hierarchy_recovery_captures.pt','implementation':ROOT/'scripts/run_bilin18_hierarchy_recovery.py','structure':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};saved=torch.load(paths['fits'],weights_only=False);B=saved['teacher'].double();torch.manual_seed(697)
 seeds=torch.tensor(list(itertools.product([0.,1.],repeat=7)),dtype=torch.float64).repeat_interleave(32,0);g=implication_closure(seeds,PARENTS);q=g.repeat_interleave(2,-1);z=torch.randn(len(g),14,dtype=torch.float64)*q
 T=torch.eye(14,dtype=torch.float64)
 for node in range(1,7):
  for parent in ancestors(node):T[2*parent:2*parent+2,2*node:2*node+2]=torch.randn(2,2,dtype=torch.float64)*.5
 S=torch.eye(14,dtype=torch.float64)
 for node in range(1,7):S[2*PARENTS[node][0]:2*PARENTS[node][0]+2,2*node:2*node+2]=.8*torch.eye(2,dtype=torch.float64)
 cfg=dict(experiment='H697',hashes=hashes,design='Construct invertible ancestor-to-child decoder shear Dnew=D T and inverse coordinate transform on all128seed patterns,32Gaussian samples each. Compare orthogonal teacher vs deliberately nonorthogonal teacher D=B S. Residualize each node against its ancestor span with exact inverse code transform. Apply same posthoc convention to H696 fitted representations.',limitations='Known hierarchy; posthoc gated-coordinate transform, not retrained linear encoder. Orthogonalization defines a convention and is not evidence nature uses orthogonal refinements. Synthetic examples, not LLM causal identification.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[]
 def assess(D,code,trueD,truecode,bias=0):
  pred=code@D.T+bias;truth=truecode@trueD.T;effects=[]
  for node in range(7):
   effect=code[:,2*node:2*node+2]@D[:,2*node:2*node+2].T;target=truecode[:,2*node:2*node+2]@trueD[:,2*node:2*node+2].T;effects.append(float((effect-target).square().sum()/target.square().sum()))
  return dict(reconstruction_relative_error=float((pred-truth).square().sum()/truth.square().sum()),component_effect_relative_errors=effects)
 for kind,D in [('orthogonal',B),('nonorthogonal',B@S)]:
  changed=D@T;zc=z@torch.linalg.inv(T).T;assert float((zc*(1-q)).abs().max())<1e-12
  canonical,A=residualize(changed);canoncode=zc@torch.linalg.inv(A).T;before=assess(changed,zc,D,z);after=assess(canonical,canoncode,D,z);assert before['reconstruction_relative_error']<1e-20 and after['reconstruction_relative_error']<1e-20
  rec=dict(kind=kind,before=before,after=after,max_valid_support_leakage=float((canoncode*(1-q)).abs().max()));records.append(rec);print(rec,flush=True)
 # Invalid child-only support cannot retain inverse-transformed ancestors under masking.
 invalid=torch.zeros(32,14,dtype=torch.float64);invalid[:,6:8]=torch.randn(32,2,dtype=torch.float64);invalidq=(invalid!=0).double();bad=(invalid@torch.linalg.inv(T).T)*invalidq;invaliderror=float((bad@(B@T).T-invalid@B.T).square().sum()/(invalid@B.T).square().sum());assert invaliderror>1e-4
 learned=[];hx,hs,hg,hc=saved['heldout'];hx,hs,hg,hc=[a.double() for a in [hx,hs,hg,hc]]
 for seed in [31,32,33]:
  m=HierarchyFit('taxonomy').double();m.load_state_dict(saved['states']['taxonomy',seed]);D=torch.nn.functional.normalize(m.decoder.dictionary,dim=1).permute(1,0,2).reshape(20,14).detach();code=(m.encoder(hx)*hg.repeat_interleave(2,-1)).detach();newD,A=residualize(D);newcode=code@torch.linalg.inv(A).T;before=assess(D,code,B,hc,m.bias.detach());after=assess(newD,newcode,B,hc,m.bias.detach());preservation=float((newcode@newD.T-code@D.T).square().sum()/(code@D.T).square().sum());assert preservation<1e-20
  root_effect=newcode[:,:2]@newD[:,:2].T;root_target=hc[:,:2]@B[:,:2].T;merged_error=float((root_effect+m.bias.detach()-root_target).square().sum()/root_target.square().sum());rec=dict(seed=seed,before=before,after=after,reconstruction_preservation_relative_error=preservation,bias_norm=float(m.bias.detach().norm()),root_effect_error_with_bias_assigned_to_always_active_root=merged_error);learned.append(rec);print(rec,flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},all_seed_patterns_checked=128,invalid_child_only_relative_error=invaliderror,finite=all(torch.isfinite(torch.tensor(r['after']['component_effect_relative_errors'])).all().item() for r in records+learned),whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'];(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,learned=learned,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
