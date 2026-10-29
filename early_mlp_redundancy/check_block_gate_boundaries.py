"""M110 controlled switching, continuous tie, no crossing, and dense-grid tests."""
import json
import numpy as np
from fit_pca_encoder import V,sha
from block_gate_boundaries import boundaries,support,decoded


def main():
    a=np.array([[1.],[0.]]);b=np.array([[-1.],[1.]]);D=np.eye(2)
    rr,ties=boundaries(a,b,1);assert not ties and len(rr)==1 and abs(rr[0]['t']-.5)<1e-12
    r=rr[0];jump=decoded(a,b,.5,D,r['right'])-decoded(a,b,.5,D,r['left']);assert np.allclose(jump,[-.5,.5])
    errors=[]
    for eps in [.01,.001,.0001]:
        finite=decoded(a,b,.5+eps,D,r['right'])-decoded(a,b,.5-eps,D,r['left']);errors.append(float(np.linalg.norm(finite-jump)))
    assert np.allclose(np.array(errors[:-1])/errors[1:],10)
    # Same selection crossing can be continuous if the decoder contributions agree.
    shared=np.ones((1,2));assert np.allclose(decoded(a,b,.5,shared,r['right']),decoded(a,b,.5,shared,r['left']))
    assert not boundaries(np.array([[2.],[.1]]),np.array([[0.],[.1]]),1)[0]
    rng=np.random.default_rng(11001);checked=0
    for trial in range(30):
      a=rng.normal(size=(6,3));b=rng.normal(size=(6,3));rr,ties=boundaries(a,b,2);assert not ties
      for r in rr:
        for sign,name in [(-1,'left'),(1,'right')]:assert list(support(a,b,r['t']+sign*r['radius']*.1,2))==r[name]
      grid=np.linspace(.000001,.999999,2001)
      actual=[support(a,b,t,2) for t in grid]
      for j in range(len(grid)-1):
        if actual[j]!=actual[j+1]:assert any(grid[j]<=r['t']<=grid[j+1] for r in rr)
      checked+=len(rr)
    out=dict(status='passed',source_sha256=sha(__file__),helper_sha256=sha(V.parent/'block_gate_boundaries.py'),random_paths=30,boundaries_checked=checked,toy_jump_norm=float(np.linalg.norm(jump)),finite_limit_errors=errors)
    dest=V/'block_gate_boundary_toys.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
