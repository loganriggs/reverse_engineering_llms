"""Polarization and simultaneous quadratic diagonalization; CPU-only control."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def recover_forms(values, d):
    q = np.zeros((2, d, d))
    for i in range(d):
        q[:, i, i] = values[i]
    k = d
    for i in range(d):
        for j in range(i + 1, d):
            q[:, i, j] = q[:, j, i] = values[k] - .5*(values[i]+values[j])
            k += 1
    return q


def alignment(a, b):
    aa = a / np.linalg.norm(a, axis=1, keepdims=True)
    bb = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos = np.abs(aa @ bb.T)
    i,j = linear_sum_assignment(-cos)
    return float(np.min(cos[i,j])), float(np.mean(cos[i,j]))


def main():
    rows, controls = [], []
    error = dict(forward=0., exact_forms=0., recovered_forms=0.)
    for d in (4,8):
        probes = list(np.eye(d))
        for i in range(d):
            for j in range(i+1,d):
                probes.append((np.eye(d)[i]+np.eye(d)[j])/np.sqrt(2))
        probes = np.array(probes)
        np.testing.assert_allclose(np.linalg.norm(probes,axis=1),1,atol=1e-14)
        for condition in (1.,3.):
            for seed in range(20):
                rng = np.random.default_rng(np.random.SeedSequence([283000+seed,d,int(condition)]))
                u,_ = np.linalg.qr(rng.normal(size=(d,d)))
                v,_ = np.linalg.qr(rng.normal(size=(d,d)))
                b = u @ np.diag(np.geomspace(1,condition,d)) @ v.T
                lam = np.linspace(-1,1,d)
                truth = np.stack([b.T@b,b.T@np.diag(lam)@b])
                h = probes @ b.T
                clean = np.column_stack([(h*h).sum(axis=1),(h*h)@lam])
                literal = np.einsum('ni,kij,nj->nk',probes,truth,probes)
                error['forward'] = max(error['forward'],float(np.max(np.abs(clean-literal))))
                np.testing.assert_allclose(clean,literal,atol=1e-12)
                exact = recover_forms(clean,d)
                error['exact_forms'] = max(error['exact_forms'],float(np.max(np.abs(exact-truth))))
                np.testing.assert_allclose(exact,truth,atol=1e-12)
                test = rng.normal(size=(512,d)); test /= np.linalg.norm(test,axis=1,keepdims=True)
                targets = np.column_stack([((test@b.T)**2).sum(axis=1),((test@b.T)**2)@lam])
                eps = rng.normal(size=clean.shape)
                # A fixed nontrivial rotation is an explicit indistinguishable alternative.
                o = np.eye(d); angle=.6
                o[:2,:2] = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
                alternative = o @ b
                repeated = lam.copy(); repeated[1]=repeated[0]
                ref_error = float(np.max(np.abs(alternative.T@alternative-truth[0])))
                repeat_error = float(np.max(np.abs(alternative.T@np.diag(repeated)@alternative-b.T@np.diag(repeated)@b)))
                assert ref_error < 1e-11 and repeat_error < 1e-11
                min_cos,_ = alignment(b,alternative)
                assert min_cos < .99
                controls.append(dict(d=d,condition=condition,seed=seed,reference_error=ref_error,
                                     repeated_output_error=repeat_error,alternative_min_cosine=min_cos))
                for sigma in (0.,.001,.01):
                    q = recover_forms(clean+sigma*eps,d)
                    eig,z = np.linalg.eigh(q[0])
                    row = dict(d=d,condition=condition,seed=seed,sigma=sigma,
                               interventions=len(probes),scalar_measurements=2*len(probes),
                               reference_min_eigenvalue=float(eig.min()))
                    if eig.min()<=0:
                        rows.append(dict(row,status='nonpositive_reference'))
                        continue
                    root = (z*np.sqrt(eig))@z.T
                    invroot = (z/np.sqrt(eig))@z.T
                    whitened = invroot@q[1]@invroot
                    weights, axes = np.linalg.eigh((whitened+whitened.T)/2)
                    inferred = axes.T @ root
                    reconstructed = np.stack([inferred.T@inferred,inferred.T@np.diag(weights)@inferred])
                    error['recovered_forms'] = max(error['recovered_forms'],float(np.max(np.abs(reconstructed-q))))
                    np.testing.assert_allclose(reconstructed,q,atol=1e-11,rtol=1e-11)
                    ht = test@inferred.T
                    pred = np.column_stack([(ht*ht).sum(axis=1),(ht*ht)@weights])
                    mn,mean = alignment(b,inferred)
                    if sigma==0:
                        assert mn>1-1e-10
                        np.testing.assert_allclose(pred,targets,atol=1e-10,rtol=1e-10)
                    rows.append(dict(row,status='ok',minimum_axis_cosine=mn,mean_axis_cosine=mean,
                                     test_mse=float(np.mean((pred-targets)**2))))
    assert len(rows)==240 and len(controls)==80
    summary=[]
    for d in (4,8):
        for condition in (1.,3.):
            for sigma in (0.,.001,.01):
                group=[r for r in rows if (r['d'],r['condition'],r['sigma'])==(d,condition,sigma)]
                ok=[r for r in group if r['status']=='ok']
                summary.append(dict(d=d,condition=condition,sigma=sigma,passed=len(ok),
                                    median_test_mse=float(np.median([r['test_mse'] for r in ok])) if ok else None,
                                    worst_axis_cosine=min(r['minimum_axis_cosine'] for r in ok) if ok else None))
    root=Path('results/unknown_quadratic_basis');root.mkdir(parents=True,exist_ok=False)
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
             ('unknown_quadratic_basis.py','UNKNOWN_QUADRATIC_BASIS_PLAN.md')}
    (root/'results.json').write_text(json.dumps(dict(records=rows,controls=controls,checks=error,
                                                    summary=summary,sources=sources,numpy_version=np.__version__),indent=2)+'\n')
    print(json.dumps(dict(summary=summary,checks=error),indent=2))


if __name__=='__main__':
    main()
