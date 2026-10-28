"""H939 known bilinear teacher: distinguish effect loss from intervention coverage."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha
from pathlib import Path
import torch,numpy as np,json,time
from scipy.optimize import linear_sum_assignment
S='bilin18_causal_objective_teacher'
MODES=['native','endpoints','effects','shuffled_effects'];SEEDS=[0,1,2]
class Student(torch.nn.Module):
 def __init__(self,seed):
  super().__init__();torch.manual_seed(seed);self.left=torch.nn.Parameter(torch.randn(16,4));self.right=torch.nn.Parameter(torch.randn(16,4));self.out=torch.nn.Parameter(torch.randn(4,4)*.1);self.bias=torch.nn.Parameter(torch.zeros(4))
 def forward(self,x):
  L=torch.nn.functional.normalize(self.left,dim=0);R=torch.nn.functional.normalize(self.right,dim=0);return ((x@L)*(x@R))@self.out+self.bias

def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');torch.manual_seed(9390);g=torch.Generator().manual_seed(9390);Q=torch.linalg.qr(torch.randn(16,16,generator=g,dtype=torch.double)).Q
 def sample(n,correlated):
  z=torch.randn(n,8,generator=g,dtype=torch.double);v=z+.05*torch.randn(n,8,generator=g,dtype=torch.double) if correlated else torch.randn(n,8,generator=g,dtype=torch.double);return torch.cat([z,v],1)@Q.T
 def teacher(x):
  z=x@Q;return z[:,:8:2]*z[:,1:8:2]
 def perturb(x,scale):
  d=torch.randn(x.shape,generator=g,dtype=torch.double);return d/d.norm(dim=1,keepdim=True)*scale
 tr=sample(512,True);delta=perturb(tr,1.0);y=teacher(tr);dy=teacher(tr+delta)-y
 z=tr@Q;dz=delta@Q;analytic=z[:,:8:2]*dz[:,1:8:2]+dz[:,:8:2]*z[:,1:8:2]+dz[:,:8:2]*dz[:,1:8:2];analyticerr=float((analytic-dy).abs().max());assert analyticerr<1e-12
 panels={}
 for name,corr in [('familiar',True),('decorrelated',False)]:
  x=sample(2048,corr);d=perturb(x,1.0);panels[name]=(x,d,teacher(x),teacher(x+d)-teacher(x))
 # Unseen double-dose and component-combination probes: columns of hidden Q only for evaluation.
 x=panels['decorrelated'][0];d=torch.zeros_like(x)
 for i in range(len(x)):d[i]=Q[:,i%8]+Q[:,(i*3+1)%8]
 panels['paired_component_edits']=(x,d,teacher(x),teacher(x+d)-teacher(x))
 cfg={'experiment':'H939','modes':MODES,'seeds':SEEDS,'steps':1500,'batch_size':128,'train_examples':512,'evaluation_examples_per_panel':2048,'teacher':'16D random orthogonal embedding of8 signal and8 nuisance coordinates. Four disjoint signal-coordinate products. Native training nuisance nearly duplicates signals; test breaks that correlation. Teacher factors/axes never supplied to optimizer. Student has4 bilinear products: known true rank is an explicit architectural prior.','objectives':'native MSE; endpoints: half native+intervened endpoint MSE; effects: native MSE + finite-effect MSE; shuffled: native MSE + permuted finite-effect targets. All same optimizer/steps/batches/initializations. Output weight penalty1e-5; normalized factors. No reconstruction objective or convex-hull constraint.','limits':'Synthetic mechanism probe, not LLM evidence. Effect/native losses use same teacher-output variance normalization; objectives differ and endpointbaseline sees same endpoint data. No claim losses are equivalent or architecture general.','next':'Compare native-vs-effect performance and true-pair recovery, then test geometric priors separately, retaining endpoint coverage control.'}
 paths={'runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};cfg['hashes']=hashes;(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 device='cuda' if torch.cuda.is_available() else 'cpu';tt=tr.float().to(device);dd=delta.float().to(device);yy=y.float().to(device);ee=dy.float().to(device);den=yy.square().mean();perm=torch.randperm(512,generator=torch.Generator().manual_seed(9391)).to(device)
 records=[];curves=[];saved={};graderr=0.
 # Independent directional derivative check in float64.
 m=Student(9392).double();xx=tr[:7].clone().requires_grad_(True);vv=delta[:7];v=m(xx).square().sum();grad=torch.autograd.grad(v,xx)[0];eps=1e-5;fd=(m(xx+eps*vv).square().sum()-m(xx-eps*vv).square().sum())/(2*eps);graderr=float(abs(fd-(grad*vv).sum()));assert graderr<1e-7
 for seed in SEEDS:
  for mode in MODES:
   m=Student(93900+seed).to(device);opt=torch.optim.Adam(m.parameters(),lr=.01);gen=torch.Generator().manual_seed(93900+seed)
   for step in range(1500):
    ix=torch.randint(512,(128,),generator=gen).to(device);a=m(tt[ix]);native=(a-yy[ix]).square().mean()/den
    if mode=='native':loss=native
    else:
     b=m(tt[ix]+dd[ix])
     if mode=='endpoints':loss=.5*(native+(b-yy[ix]-ee[ix]).square().mean()/den)
     else:target=ee[ix] if mode=='effects' else ee[perm[ix]];loss=native+((b-a)-target).square().mean()/den
    loss=loss+1e-5*m.out.square().sum();opt.zero_grad();loss.backward();opt.step()
    if step%100==0 or step==1499:curves.append(dict(seed=seed,mode=mode,step=step+1,loss=float(loss.detach()),native=float(native.detach())))
   m=m.cpu().double();saved[mode,seed]=m.state_dict()
   with torch.no_grad():
    L=torch.nn.functional.normalize(m.left,dim=0);R=torch.nn.functional.normalize(m.right,dim=0);GT=Q[:,:8];simL=(GT.T@L).abs();simR=(GT.T@R).abs();scores=torch.stack([torch.maximum((simL[2*k]+simR[2*k+1])/2,(simL[2*k+1]+simR[2*k])/2) for k in range(4)]);rr,cc=linear_sum_assignment(-scores.numpy());paircos=float(scores[rr,cc].mean())
    for name,(x,d,y0,e0) in panels.items():
     p=m(x);e=m(x+d)-p;ratio=lambda a,b:float(a.square().sum().sqrt()/b.square().sum().sqrt());records.append(dict(mode=mode,seed=seed,panel=name,native_error_ratio=ratio(p-y0,y0),effect_error_ratio=ratio(e-e0,e0),matched_pair_cosine=paircos))
   print('DONE',mode,seed,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'finite':all(np.isfinite(r['effect_error_ratio']) for r in records),'exact_teacher_finite_difference_max_abs':analyticerr,'student_gradient_directional_error':graderr,'models':len(saved),'original_model_forwards':0,'reserved_test_evaluations':0}
 assert audits['hashes_unchanged'] and audits['finite'];torch.save(dict(models=saved,teacher_basis=Q,train=tr,train_delta=delta,panels=panels),ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,curves=curves,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(json.dumps(dict(records=records,audits=audits),indent=2))
if __name__=='__main__':main()
