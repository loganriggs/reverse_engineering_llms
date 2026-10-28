"""H940: normalized hull factors can remove the directional constraint entirely."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha
from pathlib import Path
import numpy as np,torch,json,time
from scipy.optimize import linprog
S='bilin18_hull_factor_feasibility'
def ray(A,v):
 n,d=A.shape;eq=np.zeros((d+1,n+1));eq[:d,:n]=A.T;eq[:d,n]=-v;eq[d,:n]=1;b=np.r_[np.zeros(d),1.];c=np.r_[np.zeros(n),-1.];sol=linprog(c,A_eq=eq,b_eq=b,bounds=[(0,None)]*n+([(0,0)] if np.linalg.norm(v)==0 else [(0,None)]),method='highs')
 if not sol.success:return dict(success=False,status=sol.status,message=sol.message),None
 w=sol.x[:-1];t=sol.x[-1];err=max(float(np.max(np.abs(eq@sol.x-b))),float(max(0,-w.min())));dual=np.asarray(sol.eqlin.marginals);gap=abs(float(sol.fun-b@dual));assert err<1e-7 and gap<1e-7
 return dict(success=True,radius=float(t),equality_error=err,duality_gap=gap),w

def main():
 start=time.time();paths={'runner':Path(__file__),'teacher':ART/'bilin18_causal_objective_teacher_fits.pt'};hashes={k:sha(p) for k,p in paths.items()};v=torch.load(paths['teacher'],weights_only=False);Q=v['teacher_basis'].numpy();X=v['train'].numpy();D=v['train_delta'].numpy();cfg={'experiment':'H940','hashes':hashes,'design':'LP maximize nonnegative ray radius t subject to A.T*w=t*v,w>=0,sum(w)=1. Audit +/- all16 teacher-basis axes; their convex hull certifies a ball around origin if every radius positive. Compare raw observations, mean-centered observations, intervention differences, and artificial5-unit background shift. This is a geometric feasibility audit, not A-SAE replication or training comparison.','interpretation':'Normalizing a hull vector to unit norm changes its constraint to a cone-direction constraint. If0 lies strictly inside full-dimensional hull, every direction is representable. Unnormalized product factors also allow output-rescaling escape unless scale/complexity are charged.','anchors':'First128 of512 training points in half0, next128 in half1, no evaluation anchors. Difference anchors are observed random interventions, not teacher factors.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');records=[];weights={};cert=[];max_replay=0.;origin=[]
 for half in [0,1]:
  sub=X[128*half:128*(half+1)];dd=D[128*half:128*(half+1)];banks={'raw':sub,'centered':sub-sub.mean(0),'differences':dd,'shifted':sub+5*Q[:,15]}
  for kind,A in banks.items():
   result,w=ray(A,np.zeros(16));origin.append(dict(half=half,kind=kind,feasible=result['success']))
   radii=[]
   for axis in range(16):
    for sign in [-1,1]:
     r,w=ray(A,sign*Q[:,axis]);records.append(dict(half=half,kind=kind,axis=axis,sign=sign,**r))
     if w is not None:
      weights[half,kind,axis,sign]=w;rad=r['radius'];radii.append(rad)
      if rad>1e-8:
       recovered=A.T@w/rad;max_replay=max(max_replay,float(np.max(np.abs(recovered-sign*Q[:,axis]))))
   complete=len(radii)==32 and min(radii)>1e-8
   cert.append(dict(half=half,kind=kind,all_axis_rays_positive=complete,inscribed_ball_lower_bound=min(radii)/4 if complete else 0.,signal_radii=[r['radius'] for r in records if r['half']==half and r['kind']==kind and r['axis']<8 and r['success']]))
 # Construct exact teacher using scaled signal axes from raw anchors and compensate output weights.
 max_function=0.;outscale=[]
 for half in [0,1]:
  A=X[128*half:128*(half+1)];factors=np.stack([A.T@weights[half,'raw',j,1] for j in range(8)],1);rad=np.array([next(r['radius'] for r in records if r['half']==half and r['kind']=='raw' and r['axis']==j and r['sign']==1) for j in range(8)]);scale=1/(rad[::2]*rad[1::2]);outscale.append(dict(half=half,output_weights=scale.tolist()))
  for name,(x,d,y,e) in v['panels'].items():
   x=x.numpy();d=d.numpy();pred=lambda z:(z@factors[:,::2])*(z@factors[:,1::2])*scale;max_function=max(max_function,float(np.max(np.abs(pred(x)-y.numpy()))),float(np.max(np.abs(pred(x+d)-pred(x)-e.numpy()))))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'normalized_axis_replay_max_abs':max_replay,'unnormalized_rescaled_teacher_replay_max_abs':max_function,'origin_tests':origin,'lp_cases':len(records),'original_model_forwards':0,'reserved_test_evaluations':0};assert all(o['feasible']==(o['kind']!='shifted') for o in origin);assert audits['hashes_unchanged'] and max_replay<1e-6 and max_function<1e-6
 torch.save(weights,ART/f'{S}_witnesses.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,certificates=cert,output_rescaling=outscale,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(json.dumps(dict(certificates=cert,output_rescaling=outscale,audits=audits),indent=2))
if __name__=='__main__':main()
