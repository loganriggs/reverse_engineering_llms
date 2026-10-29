"""Exact radial identities plus independent quartic-model evaluation checks."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from homogeneous_evaluation import (log_radial_moment, sphere_from_gaussian,
                                    gaussian_mse_from_sphere, radial_second_moment_ratio)


def moment(k):
    return math.prod(range(1, k, 2)) if k % 2 == 0 else 0


def main():
    root = Path(__file__).parent
    out = root / 'results/homogeneous_evaluation'
    out.mkdir(exist_ok=False)
    protocol = dict(seed=100000, independent_batches=16, batch_size=8192,
                    selected_models=['shared', 'shared_wide', 'separate_width'],
                    source='multitask_confirmation/shared_0.npz; optimizer0; task0',
                    degree=4, d=8, shifts=['identity', 'diagonal std .5 to1.5'],
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    identities=[]
    for d in [2,8,16,64,256]:
        for degree in [2,4,8,16,32]:
            # E[R**2p] = product(d,d+2,...,d+2p-2).
            exact=math.prod((d+2*j)/d for j in range(degree))
            assert math.isclose(math.exp(log_radial_moment(d,2*degree)),exact,rel_tol=2e-12)
            # x1**p: multiply exact angular moment by exact radial moment.
            angular=moment(2*degree)/math.prod(d+2*j for j in range(degree))
            assert math.isclose(exact*d**degree*angular,moment(2*degree),rel_tol=1e-12)
            identities.append(dict(d=d,degree=degree,radial_second_moment_ratio=radial_second_moment_ratio(d,degree)))
    ev=np.load(root/'results/multitask_confirmation/shared_0.npz')
    keys=ev['keys']; powers=keys[:,None,:]+keys[None,:,:]
    H=np.prod(np.array([moment(k) for k in range(9)])[powers],axis=2)
    rng=np.random.default_rng(100000);rows=[]
    # Evaluate network factors directly, not the coefficient machinery being checked.
    def forward(x,A,B,W):
        z=(x@A.T)*(x@B.T);i,j=np.triu_indices(A.shape[0])
        return (z[:,i]*z[:,j])@W
    for shift in ['identity','diagonal']:
        std=np.ones(8) if shift=='identity' else np.linspace(.5,1.5,8)
        monscale=np.prod(std[None,:]**keys,axis=1)
        teacher=lambda x:forward(x,ev['teacher_A'][0],ev['teacher_B'][0],ev['teacher_W'][0])
        for model in protocol['selected_models']:
            prefix='256|0|'+model
            a,b,w=(ev[prefix+'|'+k] for k in ['A','B','W'])
            if model=='separate_width':a,b=a[0],b[0]
            w=w[0];delta=(ev[prefix+'|coefficients'][:,0]-ev['teacher_coefficients'][:,0])*monscale
            truth=float(delta@H@delta);direct=[];integrated=[];tail=[]
            for batch in range(16):
                x=rng.normal(size=(8192,8));u=sphere_from_gaussian(x)
                ordinary=(forward(x*std,a,b,w)-teacher(x*std))**2
                angular=(forward(u*std,a,b,w)-teacher(u*std))**2
                est,_=gaussian_mse_from_sphere(angular,8,4)
                direct.append(float(ordinary.mean()));integrated.append(float(est))
                tail.append(float(np.sort(ordinary)[-82:].sum()/ordinary.sum()))
            se=np.std(integrated,ddof=1)/4
            assert abs(np.mean(integrated)-truth)<max(6*se,.025*truth)
            rows.append(dict(model=model,shift=shift,exact=truth,ordinary_batches=direct,integrated_batches=integrated,
                             ordinary_top_one_percent_loss_fraction=tail,
                             integrated_mean=float(np.mean(integrated)),integrated_batch_sd=float(np.std(integrated,ddof=1)),
                             ordinary_mean=float(np.mean(direct)),ordinary_batch_sd=float(np.std(direct,ddof=1))))
    result=dict(status='passed',analytic_identities=identities,quartic_checks=rows)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:print(r['model'],r['shift'],'exact',r['exact'],'radial integrated',r['integrated_mean'],'SD ratio',r['ordinary_batch_sd']/r['integrated_batch_sd'])


if __name__=='__main__':main()
