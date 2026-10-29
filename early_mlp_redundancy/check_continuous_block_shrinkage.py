"""M111 independent formula, budget, continuity, symmetry, gradient checks."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from continuous_block_shrinkage import block_shrinkage


def reference(a,k):
    r=np.linalg.norm(a,axis=-1)
    if k==0:return np.zeros_like(a)
    if k==a.shape[-2]:return a.copy()
    threshold=np.sort(r,axis=-1)[...,-k-1,None]
    multiplier=np.maximum(r-threshold,0)/np.where(r>0,r,1)
    return a*multiplier[...,None]


def main():
    torch.set_num_threads(1);torch.manual_seed(11101);rng=np.random.default_rng(11101)
    errors=[];gradchecks=0;max_lipschitz=0.
    for shape in [(32,4),(7,32,4),(2,3,9,2)]:
      for k in [0,1,shape[-2]//2,shape[-2]-1,shape[-2]]:
        a=torch.randn(shape,dtype=torch.float64);actual=block_shrinkage(a,k);ref=reference(a.numpy(),k)
        err=float(np.max(np.abs(actual.numpy()-ref)));assert err<1e-12;errors.append(err)
        assert bool(((actual.norm(dim=-1)>0).sum(-1)<=k).all())
        assert torch.allclose(block_shrinkage(-a,k),-actual)
        assert torch.allclose(block_shrinkage(a*3.2,k),actual*3.2)
        assert torch.allclose(block_shrinkage(actual,k),actual) # <=k nonzero blocks => threshold zero
        b=torch.randn_like(a);ratio=float((block_shrinkage(a,k)-block_shrinkage(b,k)).norm()/(a-b).norm());assert ratio<=1+np.sqrt(shape[-2])+1e-12;max_lipschitz=max(max_lipschitz,ratio)
    for k in [1,2,4]:
      a=torch.randn(5,3,dtype=torch.float64,requires_grad=True)
      assert torch.autograd.gradcheck(lambda x:block_shrinkage(x,k),(a,),eps=1e-6,atol=1e-5,rtol=1e-4);gradchecks+=1
    # Group permutation and within-group orthogonal coordinates do not change the map.
    a=torch.randn(32,4,dtype=torch.float64);perm=torch.randperm(32);Q=torch.linalg.qr(torch.randn(4,4,dtype=torch.float64)).Q
    assert torch.allclose(block_shrinkage(a[perm]@Q,8),block_shrinkage(a,8)[perm]@Q,atol=1e-12)
    edge_checks=[]
    for desc,a in [('zero',torch.zeros(32,4,dtype=torch.float64)),('equal_norms',torch.ones(32,4,dtype=torch.float64)),('partly_zero',torch.cat([torch.ones(3,4,dtype=torch.float64),torch.zeros(29,4,dtype=torch.float64)]))]:
      a.requires_grad_();y=block_shrinkage(a,8);(y*y).sum().backward();assert torch.isfinite(y).all() and torch.isfinite(a.grad).all();assert ((y.norm(dim=-1)>0).sum()<=8)
      if desc=='equal_norms':assert y.count_nonzero()==0
      if desc=='partly_zero':assert torch.equal(y,a)
      edge_checks.append(dict(case=desc,active=int((y.norm(dim=-1)>0).sum()),finite_gradient=True))
    # Former hard top1 jump at [.5,.5]: shrinkage has a unique zero-valued limit.
    toy=[]
    for h in [.01,.001,.0001, .00001]:
      left=torch.tensor([[.5+h],[.5-h]],dtype=torch.float64);right=left.flip(0)
      soft=float((block_shrinkage(right,1)-block_shrinkage(left,1)).norm());hard=float(torch.tensor([.5+h,.5+h]).norm())
      assert np.isclose(soft,2*np.sqrt(2)*h,rtol=1e-9);toy.append(dict(h=h,soft_change=soft,hard_change=hard))
    # Threshold differentiation is genuinely nonlocal: an unselected threshold block affects active outputs.
    a=torch.tensor([[3.],[2.],[1.]],dtype=torch.float64,requires_grad=True)
    block_shrinkage(a,1).sum().backward();assert torch.equal(a.grad,torch.tensor([[1.],[-1.],[0.]],dtype=torch.float64))
    result=dict(status='passed',source_sha256=sha(__file__),implementation_sha256=sha(V.parent/'continuous_block_shrinkage.py'),independent_formula_cases=len(errors),max_formula_error=max(errors),gradchecks=gradchecks,edge_cases=edge_checks,toy=toy,threshold_gradient=a.grad.tolist(),max_sample_lipschitz_ratio=max_lipschitz,scope='Tests plus documented continuity argument; finite random checks alone do not prove global continuity. Derivatives at ties may be nonunique; only finite gradients are required there.')
    dest=V/'continuous_block_shrinkage_checks.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
