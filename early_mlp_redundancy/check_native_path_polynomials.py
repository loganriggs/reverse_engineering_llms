"""M114 independent direct-forward tests of rational path and quartic identities."""
import json
import numpy as np
from fit_pca_encoder import V,sha
from native_path_polynomials import path_coefficients,value,energies,crossing_intervals,select


def main():
    rng=np.random.default_rng(11401);maximum=0.;crossings=0
    for trial in range(20):
      z0,z1=rng.normal(size=(2,7));L,R=rng.normal(size=(2,14,7));D=rng.normal(size=(7,14));E=rng.normal(size=(12,7));mu=rng.normal(size=7);bias=rng.normal(size=12);scale=1.7
      u,v,q,w,n=path_coefficients(z0,z1,L,R,D,E,mu,scale,bias,np.sqrt(7));n=n.reshape(3,6,2)
      for t in np.linspace(0,1,17):
        z=u+t*(v-u);z=z/np.linalg.norm(z)*np.sqrt(7);native=D@((L@z)*(R@z));raw=E@((native-mu)/scale)+bias
        err=max(np.max(np.abs(value(w,t)/value(q,t)-native)),np.max(np.abs((value(n,t)/value(q,t)).reshape(-1)-raw)));maximum=max(maximum,float(err));assert err<1e-9
        en=energies(n);assert np.allclose(value(en.T,t),np.square(value(n,t)).sum(-1),rtol=1e-10,atol=1e-8)
      rr,ties=crossing_intervals(n,2);assert not ties
      for r in rr:
        assert list(select(n,r['t']-.1*r['radius'],2))==r['left'];assert list(select(n,r['t']+.1*r['radius'],2))==r['right']
      grid=np.linspace(0,1,1001);ss=[select(n,t,2) for t in grid]
      for i in range(len(grid)-1):
        if ss[i]!=ss[i+1]:assert any(grid[i]<=r['t']<=grid[i+1] for r in rr)
      crossings+=len(rr)
    result=dict(status='passed',source_sha256=sha(__file__),helper_sha256=sha(V.parent/'native_path_polynomials.py'),random_systems=20,direct_forward_points=340,crossings_checked=crossings,max_direct_error=maximum)
    dest=V/'native_path_polynomial_toys.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
