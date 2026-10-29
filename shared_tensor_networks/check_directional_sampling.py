"""Exact-moment controls and repeated deep energy evaluation (no training)."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from directional_sampling import DirectionalMixture,fit_axes
from deep_multitask import DeepMultiTask
from homogeneous_evaluation import sphere_from_gaussian


def main():
    torch.set_num_threads(1)
    root=Path(__file__).parent;out=root/'results/directional_sampling';out.mkdir(exist_ok=False)
    protocol=dict(batches=12,n=8192,calibration_seed=101000,pilot_seed=110000,evaluation_seed=111000,
                  pilot_directions=16,optimization_steps=80,uniform_fraction=.2,stretches=[16,64],
                  exact_controls='unit-sphere x1**(2p), d8p4/d16p16/d16p32; E_q[1/q]=1',
                  deep='d16h8 shared teacher6tasks, depths2–5; teacher energy and INITIAL student error; no trained-model comparison',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    rng=np.random.default_rng(111000);controls=[];deep=[]
    for d,p in [(8,4),(16,16),(16,32)]:
        axis=np.eye(d)[0:1];proposal=DirectionalMixture(axis)
        exact=math.exp(math.lgamma(p+.5)-math.lgamma(.5)+math.lgamma(d/2)-math.lgamma(d/2+p))
        values=[];normal=[];uniform=[]
        for batch in range(12):
            u,w=proposal.sample(8192,rng);values.append(float(np.mean(w*u[:,0]**(2*p))/exact));normal.append(float(w.mean()))
            z=rng.normal(size=(8192,d));z/=np.linalg.norm(z,axis=1,keepdims=True);uniform.append(float(np.mean(z[:,0]**(2*p))/exact))
        assert abs(np.mean(values)-1)<max(.025,6*np.std(values,ddof=1)/np.sqrt(12))
        assert abs(np.mean(normal)-1)<.025
        # Independent general-matrix inverse/determinant density formula.
        logs=[]
        for tau in (16.,64.):
            S=np.eye(d)+tau*axis.T@axis
            logs.append(np.linalg.det(S)**(-.5)*np.einsum('ni,ij,nj->n',u,np.linalg.inv(S),u)**(-d/2))
        density=.2+.4*logs[0]+.4*logs[1]
        np.testing.assert_allclose(np.exp(proposal.log_density(u)),density,atol=1e-10,rtol=1e-11)
        controls.append(dict(d=d,degree=p,weighted_relative_means=values,uniform_relative_means=uniform,normalization=normal))
        print('Exact control',d,p,'weighted mean',np.mean(values),'uniform mean',np.mean(uniform),flush=True)
    cal=torch.from_numpy(sphere_from_gaussian(np.random.default_rng(101000).normal(size=(4096,16))))
    for depth in [2,3,4,5]:
        teacher=DeepMultiTask(16,8,6,depth,True,102000+depth);teacher.calibrate(cal)
        student=DeepMultiTask(16,8,6,depth,True,103000);student.calibrate(cal)
        pilot=torch.from_numpy(np.random.default_rng(110000+depth).normal(size=(16,16)))
        axes=fit_axes(teacher,student,pilot);proposal=DirectionalMixture(axes)
        np.save(out/f'axes_depth{depth}.npy',axes)
        rows=[]
        for batch in range(12):
            for method in ['uniform','importance']:
                if method=='importance':u,w=proposal.sample(8192,rng)
                else:
                    u=rng.normal(size=(8192,16));u/=np.linalg.norm(u,axis=1,keepdims=True);w=np.ones(8192)
                with torch.no_grad():
                    x=torch.from_numpy(u*4);y=teacher(x);pred=student(x)
                    a=np.column_stack([y.square().mean(dim=1).numpy(),(pred-y).square().mean(dim=1).numpy()])*w[:,None]
                rows.append(dict(batch=batch,method=method,means=a.mean(axis=0).tolist(),ess=(a.sum(axis=0)**2/(a*a).sum(axis=0)).tolist(),weight_mean=float(w.mean()),top_one_percent=(np.sort(a,axis=0)[-82:].sum(axis=0)/a.sum(axis=0)).tolist()))
        deep.append(dict(depth=depth,rows=rows))
        for method in ['uniform','importance']:
            means=np.array([r['means'] for r in rows if r['method']==method]);ess=np.array([r['ess'] for r in rows if r['method']==method])
            print('Depth',depth,method,'means',means.mean(axis=0),'batch CV',means.std(axis=0,ddof=1)/means.mean(axis=0),'median ESS',np.median(ess,axis=0),flush=True)
    (out/'results.json').write_text(json.dumps(dict(status='complete',controls=controls,deep=deep),indent=2)+'\n')


if __name__=='__main__':main()
