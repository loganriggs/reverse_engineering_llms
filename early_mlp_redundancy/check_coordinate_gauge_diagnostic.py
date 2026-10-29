"""Independent symmetric-tensor checks and limiting gauge toys."""
import json
import numpy as np
from coordinate_gauge_diagnostic import rotation, components, summaries
from native_mixed_quadratic import prediction
from fit_pca_encoder import V, sha


def main():
    rng = np.random.default_rng(10300)
    errors = []
    for case in range(12):
        L, R, D = rng.normal(size=(7,5)), rng.normal(size=(7,5)), rng.normal(size=(3,7))
        T = np.einsum('on,ni,nj->oij', D,L,R)
        T = (T+T.swapaxes(1,2))/2
        F = lambda x: np.einsum('...i,oij,...j->...o', x,T,x)
        z, delta, mean = rng.normal(size=(3,5))
        B = np.linalg.qr(rng.normal(size=(5,3)))[0]
        original = None
        for Q in [np.eye(3), rotation(3,10301),rotation(3,10302),rotation(3,10303)]:
            basis = B@Q
            parts, mid, single, response, interaction = components(L,R,D,z,delta,basis)
            assert np.allclose(single,F(z+parts)-F(z),atol=1e-10,rtol=1e-10)
            assert np.allclose(mid.sum(0),response,atol=1e-10,rtol=1e-10)
            assert np.allclose(single.sum(0)+interaction,response,atol=1e-10,rtol=1e-10)
            mixed = prediction(L,R,D,mean,basis,basis,z)
            if original is None: original = (response,mixed,parts.sum(0))
            for a,b in zip((response,mixed,parts.sum(0)),original):
                assert np.allclose(a,b,atol=1e-10,rtol=1e-10)
                errors.append(float(np.linalg.norm(a-b)))
    # The same linear computation changes from one participating axis to eight.
    k=8;delta=np.ones(k)/np.sqrt(k);J=delta[None,:]
    aligned=np.linalg.qr(np.column_stack([delta,np.eye(k)[:,1:]]))[0]
    toys=[]
    for name,B in [('aligned',aligned),('spread',np.eye(k))]:
        parts=(delta@B)[:,None]*B.T;out=parts@J.T
        toys.append(dict(basis=name,participation=summaries(parts,out,out,out.sum(0),np.zeros(1))['midpoint_participation']))
    assert np.isclose(toys[0]['participation'],1) and np.isclose(toys[1]['participation'],8)
    # F=x*y: axis-at-a-time changes vanish at zero, but the combined change is 1.
    L=np.array([[1.,0.]]);R=np.array([[0.,1.]]);D=np.ones((1,1));z=np.zeros(2);delta=np.ones(2)
    rows=[]
    for name,B in [('axes',np.eye(2)),('diagonal',np.array([[1.,1.],[1.,-1.]])/np.sqrt(2))]:
        c=components(L,R,D,z,delta,B);rows.append(dict(basis=name,**summaries(*c)))
    assert np.isclose(rows[0]['interaction_relative_energy'],1) and rows[1]['interaction_relative_energy']<1e-20
    result=dict(status='passed',source_sha256=sha(__file__),helper_sha256=sha(V.parent/'coordinate_gauge_diagnostic.py'),random_tensor_cases=12,max_invariance_absolute_error=max(errors),linear_localized_to_distributed=toys,quadratic_interaction_gauge=rows,scope='Tensor oracle is independent of product-factor implementation. Same function/span can have different coordinate participation and finite-intervention interaction remainder. These are coordinate diagnostics, not native-channel deletion or semantic tests.')
    dest=V/'coordinate_gauge_toys.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
