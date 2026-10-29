"""Independent autograd/finite-difference verification of readout elimination."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from variable_projection import reduced_residual

torch.set_default_dtype(torch.float64)
rows=[]
for d,h in [(3,2),(6,3),(8,4)]:
 for ridge in [0., .001]:
    rng=np.random.default_rng(231000+10*d+h)
    x=rng.normal(size=(80,d)); y=rng.normal(size=(80,3))
    theta=rng.normal(size=2*h*d)/np.sqrt(d)
    xt,yt=torch.tensor(x),torch.tensor(y)
    pairs=list(itertools.combinations_with_replacement(range(h),2));p=len(pairs)
    def residual(t):
        a,b=t[:h*d].reshape(h,d),t[h*d:].reshape(h,d)
        z=(xt@a.T)*(xt@b.T)
        phi=torch.stack([z[:,i]*z[:,j] for i,j in pairs],dim=1)
        Z=torch.cat([phi,np.sqrt(ridge)*torch.eye(p)])
        Y=torch.cat([yt,torch.zeros(p,3)])
        U=torch.linalg.solve(Z.T@Z,Z.T@Y)
        return Z@U-Y
    t=torch.tensor(theta,requires_grad=True)
    r,J,U=reduced_residual(theta,x,y,h,ridge)
    rt=residual(t).detach().numpy()
    jt=torch.autograd.functional.jacobian(residual,t).numpy()
    rel=np.linalg.norm(J-jt)/max(1.,np.linalg.norm(jt))
    assert rel<1e-8,rel
    np.testing.assert_allclose(r,rt,atol=1e-9,rtol=1e-9)
    direction=rng.normal(size=len(t));direction/=np.linalg.norm(direction)
    eps=1e-5
    rp=reduced_residual(theta+eps*direction,x,y,h,ridge)[0]
    rm=reduced_residual(theta-eps*direction,x,y,h,ridge)[0]
    fd=(rp-rm)/(2*eps); exact=J@direction
    fdrel=np.linalg.norm(fd-exact)/max(1.,np.linalg.norm(exact))
    assert fdrel<1e-7,fdrel
    # Envelope gradient: freeze the optimum readout and differentiate only Phi.
    from direct_bilinear_readout import predict_jacobian
    _,fullJ=predict_jacobian(np.r_[theta,U.T.ravel()],x,h)
    envelope=np.einsum('nt,ntk->k',r[:len(x)],fullJ[:,:,:len(theta)])
    actual=np.einsum('nt,ntk->k',r,J)
    np.testing.assert_allclose(envelope,actual,atol=1e-8,rtol=1e-8)
    rows.append(dict(d=d,h=h,ridge=ridge,jacobian_relative_error=rel,
                     finite_difference_relative_error=fdrel,
                     envelope_error=float(np.max(np.abs(actual-envelope)))))
out=Path('results/variable_projection');out.mkdir(exist_ok=True)
(out/'checks.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
