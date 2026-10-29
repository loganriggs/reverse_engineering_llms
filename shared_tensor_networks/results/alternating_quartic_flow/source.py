"""Exact frozen-block gradient flow; eigen-exponential solves of quadratic block dynamics."""
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from assess_fixed_degree_recovery import assess


def gradient(a, x, w, target, scale, block=None):
    values = x @ a.T
    residual = values.prod(1)/scale-target
    blocks = range(4) if block is None else [block]
    gradients = [x.T @ (w*residual*np.prod(values[:,[j for j in range(4) if j!=i]],axis=1)/scale) for i in blocks]
    return np.stack(gradients) if block is None else gradients[0]


def step(a, x, w, target, scale, lr, order):
    a = a.copy()
    for i in order:
        values = x @ a.T
        multiplier = np.prod(values[:,[j for j in range(4) if j!=i]],axis=1)/scale
        design = x*multiplier[:,None]
        hessian = design.T @ (w[:,None]*design)
        rhs = design.T @ (w*target)
        eigenvalues,vectors = np.linalg.eigh(hessian)
        assert eigenvalues.min() >= -1e-12*max(1,np.linalg.norm(hessian))
        eigenvalues = np.maximum(eigenvalues,0.)
        phi = np.full_like(eigenvalues,lr)
        positive = eigenvalues>0
        phi[positive] = -np.expm1(-lr*eigenvalues[positive])/eigenvalues[positive]
        a[i] = vectors @ (np.exp(-lr*eigenvalues)*(vectors.T@a[i])+phi*(vectors.T@rhs))
    return a


def main():
    root = Path(__file__).parent
    out = root/'results/alternating_quartic_flow'
    out.mkdir(exist_ok=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    z,w1 = np.polynomial.hermite_e.hermegauss(5)
    ids = np.array(list(itertools.product(range(5),repeat=3)))
    x,w = z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1)
    protocol = dict(seeds=[170001,170002,170003],rates=[.03,.01,.003],time=30,
        modes=['forward','reverse'],initialization='archived exactly symmetric four factors',
        objective='half Gaussian population MSE;125point exact quadrature',
        clock='each factor receives one eta update per sweep; final block time30',
        work='4 exact3x3 affine ODE solves/sweep; frozen blocks evolve for eta; not equal FLOPs to gradient steps',
        source_sha256=sha(Path(__file__)),recovery_dependency_sha256=sha(root/'assess_fixed_degree_recovery.py'))
    pp=out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    for seed in protocol['seeds']:
        original=root/f'results/fixed_degree_reuse/t4_s{seed}_symmetric_g4.npz'
        data=np.load(original);initial=data['initial'];teacher=data['teacher']
        scale=float(data['student_scale']);tscale=float(data['teacher_scale']);target=(x@teacher.T).prod(1)/tscale
        for lr in protocol['rates']:
            for mode in protocol['modes']:
                name=f's{seed}_lr{lr}_{mode}';dest=out/f'{name}.json'
                if dest.exists():continue
                a=initial.copy();rng=np.random.default_rng(seed+201005);curves=[];failure=None;start=time.monotonic()
                steps=round(30/lr)
                for k in range(steps+1):
                    if k in {0,1,round(1/lr),round(3/lr),round(10/lr),steps}:
                        mse=float(w@((x@a.T).prod(1)/scale-target)**2)
                        curves.append(dict(sweep=k,time=k*lr,population_mse=mse,
                            factor_spread=float(np.linalg.norm(a-a.mean(0)))))
                    if k==steps:break
                    order=None if mode=='simultaneous' else range(4) if mode=='forward' else range(3,-1,-1) if mode=='reverse' else rng.permutation(4)
                    a=step(a,x,w,target,scale,lr,order)
                    if not np.isfinite(a).all():
                        failure=dict(sweep=k+1,reason='nonfinite parameters');break
                row=dict(name=name,seed=seed,lr=lr,mode=mode,curves=curves,failure=failure,
                    block_flow_solves=4*(k if failure is None else k+1),
                    forward_evaluations=(1 if mode=='simultaneous' else 4)*(k if failure is None else k+1),
                    seconds=time.monotonic()-start,input_sha256=sha(original),
                    recovery=assess(teacher,a,tscale,scale) if failure is None else None)
                np.savez_compressed(out/f'{name}.npz',teacher=teacher,teacher_scale=tscale,initial=initial,student=a,student_scale=scale)
                dest.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
                print(name,curves[-1],failure,flush=True)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',fits=18))+'\n')


if __name__=='__main__':main()
