import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_residual_feature_objective import fit_metric,metric_apply

def test_conditional_metric_solver_matches_dense_normal_equations():
    torch.manual_seed(94);f=torch.randn(20,3,dtype=torch.float64);y=torch.randn(20,4,dtype=torch.float64);g=torch.randn(20,2,4,dtype=torch.float64)
    for grads in [None,g]:
        w,diag=fit_metric(f,y,.1,grads,maxiter=200,tol=1e-10)
        x=torch.cat([f,torch.ones(20,1,dtype=f.dtype)],1);scale=4/g.square().sum(-1).mean() if grads is not None else 0;pen=torch.ones(4,1,dtype=f.dtype);pen[-1]=0
        def op(z):return x.T@metric_apply(x@z,grads,scale)/20+.1*f.square().mean()*pen*z
        eye=torch.eye(16,dtype=f.dtype);mat=torch.stack([op(v.reshape(4,4)).flatten() for v in eye],1);rhs=(x.T@metric_apply(y,grads,scale)/20).flatten();expected=torch.linalg.solve(mat,rhs).reshape(4,4)
        torch.testing.assert_close(w,expected,rtol=1e-7,atol=1e-7)
        assert diag['converged']

def test_quadratic_features_preserve_symmetric_outer_product_norm():
    from run_bilin18_l17_residual_feature_objective import raw_features
    torch.manual_seed(95);x=torch.randn(3,1152);f=raw_features(x,torch.eye(1152),torch.zeros(1152),'quadratic1152')
    assert f.shape==(3,1152)
    torch.testing.assert_close(f[:,:624],x[:,:624])
    torch.testing.assert_close(f[:,624:].square().sum(-1),x[:,:32].square().sum(-1).square())
