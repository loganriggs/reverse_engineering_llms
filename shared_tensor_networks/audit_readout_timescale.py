"""Independent literal-forward replay of saved two-timescale trajectories."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path('results/readout_timescale')


def phi(theta,x):
    raw=theta.reshape(4,3);norm=np.linalg.norm(raw,axis=1);unit=raw/norm[:,None]
    z=(x@unit[:2].T)*(x@unit[2:].T)
    return np.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(2),2)],axis=1),norm


def main():
    records=json.loads((ROOT/'results.json').read_text());assert len(records)==30
    checks=json.loads((ROOT/'checks.json').read_text());assert len(checks)==60
    replay=[];solver=[];hashes={}
    for row in records:
        data=np.load(ROOT/f'data_{row["seed"]}.npz');run=np.load(ROOT/row['trajectory'])
        np.testing.assert_allclose(phi(data['teacher'].ravel(),data['x'])[0]@data['W'],data['y'],atol=1e-12)
        losses=[];errors=[]
        for v in run['path'].T:
            f,norm=phi(v[:12],data['x']);u=v[12:].reshape(3,3)
            # Independent SVD-based ridge solve of augmented design.
            z=np.r_[f/np.sqrt(32),np.sqrt(.1)*np.eye(3)]
            y=np.r_[data['y']/np.sqrt(32),np.zeros((3,3))]
            us=np.linalg.lstsq(z,y,rcond=None)[0]
            losses.append(.5*(np.sum((z@u-y)**2)+np.sum((norm-1)**2)))
            errors.append(np.linalg.norm(u-us))
        np.testing.assert_allclose(losses,run['loss'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(errors,run['tracking'],rtol=1e-7,atol=1e-9)
        distance=float(np.linalg.norm(run['path'][:12,-1]-data['reduced'][:,-1]))
        np.testing.assert_allclose(distance,row['hidden_endpoint_distance'],rtol=1e-12)
        np.testing.assert_allclose(run['path'][:12,0],data['theta'],atol=1e-14)
        replay.append(dict(trajectory=row['trajectory'],observations=len(losses),max_loss_error=float(np.max(np.abs(np.array(losses)-run['loss'])))))
        explicit=Path('results/readout_timescale_explicit_attempt')/row['trajectory']
        if explicit.exists():
            other=np.load(explicit);error=float(np.max(np.abs(other['path']-run['path'])))
            assert error<2e-7,error
            solver.append(dict(trajectory=row['trajectory'],max_solver_discrepancy=error))
    assert len(solver)>=2
    for path in sorted(ROOT.glob('*')):
        if path.suffix in ['.npz','.json'] and path.name!='audit.json':hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(fits=len(replay),observations=sum(r['observations'] for r in replay),replay=replay,independent_solver_comparisons=solver,sha256=hashes,
                scope='Literal forward/loss and SVD tracking replay; solver comparison where explicit runs completed. Not an interval ODE certificate.')
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['replay','sha256']},indent=2))


if __name__=='__main__':main()
