"""Verify reduced-residual GN versus profiled joint-linearization GN."""
import json
from pathlib import Path
import numpy as np
from normalized_readout import objective
root=Path('results/normalized_readout/panel');rows=[]
for seed in range(232101,232106):
 data=np.load(root/f'{seed}_data.npz');theta=np.load(root/f'{seed}_233001_reduced.npz')['initial'];x,y=data['x'],data['y']
 for ridge in [0.,.001]:
  r,J,U,_=objective(theta,x,y,4,ridge=ridge)
  rj,Jj,_,_=objective(np.r_[theta,U.T.ravel()],x,y,4,ridge=ridge,reduced=False)
  np.testing.assert_allclose(r,rj,atol=1e-10)
  H=Jj[:,:64];V=Jj[:,64:];Q,R=np.linalg.qr(V,mode='reduced')
  A=H-Q@(Q.T@H);B=J-A
  cross=np.linalg.norm(A.T@B)/max(1.,np.linalg.norm(A)*np.linalg.norm(B))
  tangent=np.linalg.norm(B-Q@(Q.T@B))/max(1.,np.linalg.norm(B))
  gradient=np.linalg.norm(J.T@r-A.T@r)/max(1.,np.linalg.norm(J.T@r))
  identity=np.linalg.norm(J.T@J-A.T@A-B.T@B)/max(1.,np.linalg.norm(J.T@J))
  assert max(cross,tangent,gradient,identity)<1e-8
  damp=.001
  jointH=Jj.T@Jj+np.diag(np.r_[np.full(64,damp),np.zeros(100)])
  jointstep=np.linalg.solve(jointH,-Jj.T@r)
  profstep=np.linalg.solve(A.T@A+damp*np.eye(64),-A.T@r)
  step_error=np.linalg.norm(jointstep[:64]-profstep)/max(1.,np.linalg.norm(profstep));assert step_error<1e-7
  reducedstep=np.linalg.solve(J.T@J+damp*np.eye(64),-J.T@r)
  rows.append(dict(seed=seed,ridge=ridge,cross_error=cross,tangent_error=tangent,gradient_error=gradient,
    curvature_identity_error=identity,profiled_joint_step_error=step_error,
    extra_curvature_trace_ratio=float(np.sum(B*B)/np.sum(A*A)),
    reduced_vs_profiled_step_difference=float(np.linalg.norm(reducedstep-profstep)/np.linalg.norm(profstep))))
Path('results/crossed_readout/profiled_gn_checks.json').write_text(json.dumps(rows,indent=2)+'\n')
print('10 geometry checks passed. Extra curvature trace ratios:',[round(r['extra_curvature_trace_ratio'],3) for r in rows])
