"""H941 scale-accounted bilinear prior pilot; not an A-SAE replication."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha
from pathlib import Path
import torch,numpy as np,json,time
from scipy.optimize import linear_sum_assignment
S='bilin18_scale_accounted_priors';KINDS=['free','state_hull','difference_hull','normalized_state'];BUDGETS=[2.,32.,2048.]
class Student(torch.nn.Module):
 def __init__(self,kind,anchors,seed,budget):
  super().__init__();torch.manual_seed(94100+seed);self.kind=kind;self.budget=budget;self.register_buffer('anchors',anchors);self.left=torch.nn.Parameter(torch.randn(16 if kind=='free' else len(anchors),4)*.5);self.right=torch.nn.Parameter(torch.randn_like(self.left)*.5);self.out=torch.nn.Parameter(torch.randn(4,4)*.1);self.bias=torch.nn.Parameter(torch.zeros(4))
 def factors(self):
  if self.kind=='free':return torch.nn.functional.normalize(self.left,dim=0),torch.nn.functional.normalize(self.right,dim=0)
  L=self.anchors.T@self.left.softmax(0);R=self.anchors.T@self.right.softmax(0)
  if self.kind=='normalized_state':return torch.nn.functional.normalize(L,dim=0),torch.nn.functional.normalize(R,dim=0)
  # Shrinkage is hull preserving because origin interior was verified; also bound all free factors by1.
  return L/L.norm(dim=0).clamp_min(1.),R/R.norm(dim=0).clamp_min(1.)
 def forward(self,x):
  L,R=self.factors();return ((x@L)*(x@R))@self.out+self.bias
 def project(self):
  with torch.no_grad():self.out.mul_(min(1.,self.budget/float(self.out.norm().clamp_min(1e-12))))

def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'data':ART/'bilin18_causal_objective_teacher_fits.pt','geometry':ART/'bilin18_hull_factor_feasibility.json'};hashes={k:sha(v) for k,v in paths.items()};data=torch.load(paths['data'],weights_only=False);Q=data['teacher_basis'];x=data['train'];d=data['train_delta'];z=x@Q;zz=(x+d)@Q;y=z[:,:8:2]*z[:,1:8:2];e=zz[:,:8:2]*zz[:,1:8:2]-y
 cfg={'experiment':'H941','hashes':hashes,'kinds':KINDS,'gain_budgets':BUDGETS,'seeds':[0,1],'anchor_halves':[0,1],'steps':1500,'loss':'native MSE + finite-effect MSE, both normalized by native-output energy. No output-weight penalty; explicit Frobenius gain budget enforced after every step.','controls':'Same512 states/interventions, minibatches per initialization seed,4 products, same optimizer and steps. Free factors unitnorm; hull factors native convex combinations with norm<=1; normalized-state only optimizationcontrol. State and difference anchors rescaled to unit RMS norm separately using training data. Raw bank scales saved. Two independent128-anchor banks crossed with two initializations. Free has no anchors and is trained once per seed/budget.','limits':'Parameter counts differ (hull logits vs free factors), not convergence proof. Known teacher rank supplied. No claim that hull constraint fixes gauge or means same geometry as A-SAE. Anchored banks all span16D; their scale affects effective gain constraints.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 device='cuda';xx=x.float().to(device);dd=d.float().to(device);yy=y.float().to(device);ee=e.float().to(device);den=yy.square().mean();records=[];curves=[];models={};bankscales=[];maxnorm=0.;maxgain=0.
 for kind in KINDS:
  for half in ([-1] if kind=='free' else [0,1]):
   raw=(d if kind=='difference_hull' else x)[128*max(0,half):128*(max(0,half)+1)];scale=raw.square().sum(1).mean().sqrt();anchors=(raw/scale).float();bankscales.append(dict(kind=kind,half=half,rms_scale=float(scale)))
   for seed in [0,1]:
    for budget in BUDGETS:
     m=Student(kind,anchors.to(device),seed,budget).to(device);opt=torch.optim.Adam(m.parameters(),lr=.01);gen=torch.Generator().manual_seed(94100+seed)
     for step in range(1500):
      ii=torch.randint(len(xx),(128,),generator=gen).to(device);a=m(xx[ii]);b=m(xx[ii]+dd[ii]);native=(a-yy[ii]).square().mean()/den;effect=(b-a-ee[ii]).square().mean()/den;loss=native+effect;opt.zero_grad();loss.backward();opt.step();m.project()
      if step%250==0 or step==1499:curves.append(dict(kind=kind,half=half,seed=seed,budget=budget,step=step+1,loss=float(loss.detach())))
     m=m.cpu().double();models[kind,half,seed,budget]=m.state_dict()
     with torch.no_grad():
      L,R=m.factors();ln=L.norm(dim=0);rn=R.norm(dim=0);gain=float(m.out.norm());maxnorm=max(maxnorm,float(ln.max()),float(rn.max()));maxgain=max(maxgain,gain-budget)
      SL=(Q[:,:8].T@torch.nn.functional.normalize(L,dim=0)).abs();SR=(Q[:,:8].T@torch.nn.functional.normalize(R,dim=0)).abs();scores=torch.stack([torch.maximum((SL[2*k]+SR[2*k+1])/2,(SL[2*k+1]+SR[2*k])/2) for k in range(4)]);ri,ci=linear_sum_assignment(-scores.numpy());paircos=float(scores[ri,ci].mean())
      for name,(tx,td,ty,te) in data['panels'].items():
       p=m(tx);pe=m(tx+td)-p;ratio=lambda err,ref:float(err.square().sum().sqrt()/ref.square().sum().sqrt());records.append(dict(kind=kind,half=half,seed=seed,budget=budget,panel=name,native_error_ratio=ratio(p-ty,ty),effect_error_ratio=ratio(pe-te,te),matched_pair_cosine=paircos,factor_norm_min=float(torch.cat([ln,rn]).min()),factor_norm_max=float(torch.cat([ln,rn]).max()),output_gain=gain,trainable_parameters=sum(p.numel() for p in m.parameters())))
     print('DONE',kind,half,seed,budget,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'factor_norm_max':maxnorm,'gain_budget_max_excess':maxgain,'finite':all(np.isfinite(r['effect_error_ratio']) for r in records),'models':len(models),'original_model_forwards':0,'reserved_test_evaluations':0};assert audits['hashes_unchanged'] and maxnorm<1.000001 and maxgain<1e-4 and audits['finite'];torch.save(dict(models=models,bankscales=bankscales),ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,curves=curves,bankscales=bankscales,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
