"""H942 matched function initialization for free vs normalized-hull factors."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha
from run_bilin18_scale_accounted_priors import Student
from run_bilin18_hull_factor_feasibility import ray
from pathlib import Path
from scipy.optimize import linprog,linear_sum_assignment
import numpy as np,torch,json,time
S='bilin18_matched_hull_initialization'
def origin(A):
 n,d=A.shape;# maximize smallest weight, keeping an exact zero convex combination
 c=np.r_[np.zeros(n),-1.];eq=np.zeros((d+1,n+1));eq[:d,:n]=A.T;eq[d,:n]=1;ub=np.c_[-np.eye(n),np.ones(n)];sol=linprog(c,A_ub=ub,b_ub=np.zeros(n),A_eq=eq,b_eq=np.r_[np.zeros(d),1],bounds=[(0,None)]*(n+1),method='highs');assert sol.success and sol.x[-1]>1e-8;return sol.x[:-1]
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'student':ROOT/'scripts/run_bilin18_scale_accounted_priors.py','ray':ROOT/'scripts/run_bilin18_hull_factor_feasibility.py','data':ART/'bilin18_causal_objective_teacher_fits.pt','prior':ART/'bilin18_scale_accounted_priors.json'};hashes={k:sha(v) for k,v in paths.items()};data=torch.load(paths['data'],weights_only=False);Q=data['teacher_basis'];x=data['train'];d=data['train_delta'];z=x@Q;zz=(x+d)@Q;y=z[:,:8:2]*z[:,1:8:2];e=zz[:,:8:2]*zz[:,1:8:2]-y
 cfg={'experiment':'H942','hashes':hashes,'steps':1500,'seeds':[0,1],'anchor_halves':[0,1],'gain_cap':32.,'design':'Use exact same initial effective unit factors/output matrix/bias as H941 free model. Obtain strictly-positive simplex witnesses from half ray-maximizer plus half max-min-weight origin combination; initialize normalized-hull logits with log(weights). Original native/effect targets, optimizer and batches retained. No ground truth factors or evaluation losses enter initialization. This isolates initial function, not parameter-space conditioning or storage.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 xx=x.float().cuda();dd=d.float().cuda();yy=y.float().cuda();ee=e.float().cuda();den=yy.square().mean();models={};witnesses={};records=[];curves=[];initerr=0.;floaterr=0.;eqerr=0.
 for seed in [0,1]:
  for half in [-1,0,1]:
   raw=x[128*max(half,0):128*(max(half,0)+1)];anchors=raw/raw.square().sum(1).mean().sqrt();free=Student('free',anchors.float(),seed,32.).double();m=Student('free' if half<0 else 'normalized_state',anchors.float(),seed,32.).double();m.anchors.copy_(anchors)
   if half>=0:
    A=anchors.numpy();ow=origin(A);L,R=free.factors()
    with torch.no_grad():
     for label,F in [('left',L),('right',R)]:
      cols=[]
      for j in range(4):
       dr=F[:,j].detach().numpy();rr,rw=ray(A,dr);assert rr['success'] and rr['radius']>1e-8;w=.5*(rw+ow);assert min(w)>0;eqerr=max(eqerr,float(abs(w.sum()-1)),float(np.max(np.abs(A.T@w-.5*rr['radius']*dr))));cols.append(torch.from_numpy(w));witnesses[seed,half,label,j]=w
      getattr(m,label).copy_(torch.stack(cols,1).log())
     m.out.copy_(free.out);m.bias.copy_(free.bias)
   with torch.no_grad():initerr=max(initerr,float((m(x)-free(x)).abs().max()));m=m.float().cuda();free=free.float().cuda();floaterr=max(floaterr,float((m(xx)-free(xx)).abs().max()))
   del free
   opt=torch.optim.Adam(m.parameters(),lr=.01);gen=torch.Generator().manual_seed(94100+seed)
   for step in range(1500):
    ii=torch.randint(len(xx),(128,),generator=gen).cuda();a=m(xx[ii]);b=m(xx[ii]+dd[ii]);loss=((a-yy[ii]).square().mean()+(b-a-ee[ii]).square().mean())/den;opt.zero_grad();loss.backward();opt.step();m.project()
    if step%250==0 or step==1499:curves.append(dict(seed=seed,half=half,step=step+1,loss=float(loss.detach())))
   m=m.cpu().double();models[seed,half]=m.state_dict()
   with torch.no_grad():
    L,R=m.factors();SL=(Q[:,:8].T@L).abs();SR=(Q[:,:8].T@R).abs();sc=torch.stack([torch.maximum((SL[2*k]+SR[2*k+1])/2,(SL[2*k+1]+SR[2*k])/2) for k in range(4)]);ri,ci=linear_sum_assignment(-sc.numpy());cos=float(sc[ri,ci].mean())
    for name,(tx,td,ty,te) in data['panels'].items():
     p=m(tx);pe=m(tx+td)-p;ratio=lambda err,ref:float(err.square().sum().sqrt()/ref.square().sum().sqrt());records.append(dict(seed=seed,half=half,panel=name,native_error_ratio=ratio(p-ty,ty),effect_error_ratio=ratio(pe-te,te),matched_pair_cosine=cos))
   print('DONE',seed,half,flush=True)
 prior=json.loads(paths['prior'].read_text());baseline=max(abs(r['effect_error_ratio']-next(q['effect_error_ratio'] for q in prior['records'] if q['kind']=='free' and q['seed']==r['seed'] and q['budget']==32. and q['panel']==r['panel'])) for r in records if r['half']==-1)
 audits={'hashes_unchanged':all(sha(v)==hashes[k] for k,v in paths.items()),'initial_function_double_max_abs':initerr,'initial_function_float_max_abs':floaterr,'witness_equation_max_abs':eqerr,'baseline_replay_max_abs':baseline,'finite':all(np.isfinite(r['effect_error_ratio']) for r in records),'models':6,'original_model_forwards':0,'reserved_test_evaluations':0};assert audits['hashes_unchanged'] and initerr<1e-8 and floaterr<1e-4 and eqerr<1e-8 and baseline<.01 and audits['finite'];torch.save(dict(models=models,witnesses=witnesses),ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,curves=curves,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(records,audits,flush=True)
if __name__=='__main__':main()
