import numpy as np
from bilin18_conditional_response_basis import conditional_statistics,directions

def test_conditional_statistics_match_sample_residualization():
    rng=np.random.default_rng(8);x=rng.normal(size=(100,12));y=rng.normal(size=(100,9));q=np.linalg.qr(rng.normal(size=(12,12)))[0];a=q[:,:4];b=q[:,4:];g=x.T@x;r=x.T@y;s,t=conditional_statistics(g,r,a,b);xa=x@a;xb=x@b;resx=xb-xa@np.linalg.lstsq(xa,xb,rcond=None)[0];resy=y-xa@np.linalg.lstsq(xa,y,rcond=None)[0]
    assert np.allclose(s,resx.T@resx,atol=1e-10)
    assert np.allclose(t,resx.T@resy,atol=1e-10)
    basis,_=directions(s,t,b,.1,3)
    assert np.allclose(a.T@basis,0,atol=1e-12)
    assert np.allclose(basis.T@basis,np.eye(3),atol=1e-12)
