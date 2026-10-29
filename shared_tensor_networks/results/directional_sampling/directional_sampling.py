"""Defensive mixture of rank-one angular central Gaussian proposals."""
import numpy as np
from scipy.special import logsumexp


class DirectionalMixture:
    def __init__(self,axes,stretches=(16.,64.),uniform_fraction=.2):
        axes=np.asarray(axes,dtype=float)
        assert axes.ndim==2 and len(axes)>0 and 0<uniform_fraction<1
        axes=axes/np.linalg.norm(axes,axis=1,keepdims=True)
        self.axes=np.repeat(axes,len(stretches),axis=0)
        self.tau=np.tile(np.asarray(stretches),len(axes))
        assert np.all(self.tau>=0)
        self.d=axes.shape[1];self.floor=uniform_fraction

    def log_density(self,u):
        """Log density relative to uniform UNIT sphere surface measure."""
        u=np.asarray(u);assert np.allclose(np.linalg.norm(u,axis=1),1.,rtol=1e-10,atol=1e-10)
        dot=u@self.axes.T
        quadratic=1-dot**2*(self.tau/(1+self.tau))
        logs=-.5*np.log1p(self.tau)-self.d/2*np.log(quadratic)
        return np.logaddexp(np.log(self.floor),np.log1p(-self.floor)-np.log(len(self.axes))+logsumexp(logs,axis=1))

    def sample(self,n,rng):
        labels=rng.integers(0,len(self.axes),size=n)
        active=rng.random(n)>=self.floor
        x=rng.normal(size=(n,self.d));v=self.axes[labels]
        stretch=(np.sqrt(1+self.tau[labels])-1)*active
        x+=stretch[:,None]*np.sum(x*v,axis=1)[:,None]*v
        u=x/np.linalg.norm(x,axis=1,keepdims=True)
        weights=np.exp(-self.log_density(u))
        assert np.all(weights<=1/self.floor*(1+1e-12))
        return u,weights


def fit_axes(model,student,pilot,steps=80):
    """Optimize fixed pilot directions; no evaluation points are reused."""
    import torch
    x=pilot.detach().clone().requires_grad_(True)
    for _ in range(steps):
        u=x/torch.linalg.vector_norm(x,dim=1,keepdim=True)*np.sqrt(x.shape[1])
        y=model(u);e=student(u)-y
        objective=torch.log(y.square().mean(dim=1)+e.square().mean(dim=1)+1e-30).sum()
        grad,=torch.autograd.grad(objective,x)
        with torch.no_grad():
            x+=.15*grad/torch.linalg.vector_norm(grad,dim=1,keepdim=True).clamp_min(1e-20)
            x/=torch.linalg.vector_norm(x,dim=1,keepdim=True)
    return x.detach().numpy()
