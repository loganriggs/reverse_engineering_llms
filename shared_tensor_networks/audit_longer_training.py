"""Independent direct NumPy Gaussian quadrature for all extended-training snapshots."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from audit_results import forward
from experiment import gram
from regression_baselines import feature_maps


def main():
    root=Path(__file__).parent;out=root/'results/longer_training';src=root/'results/e1_d8'
    rows=list(csv.DictReader((out/'results.csv').open()));assert len(rows)==180
    assert len({(r['teacher'],r['seed'],r['n'],r['student'],r['step']) for r in rows})==180
    assert len(json.loads((out/'replays.json').read_text()))==60
    node,weight=np.polynomial.hermite_e.hermegauss(5);weight/=np.sqrt(2*np.pi)
    index=np.array(list(itertools.product(range(5),repeat=8)));x=node[index];weight=weight[index].prod(axis=1)
    keys=sorted({tuple(sum(i==j for i in inds) for j in range(8)) for inds in itertools.combinations_with_replacement(range(8),4)})
    H=gram(keys);L=np.linalg.cholesky(H);teachers={};errors=[];floors=[];hashes={}
    for row in rows:
        path=out/row['evidence'];hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        weights=np.load(path);identity=(row['teacher'],row['seed'])
        if identity not in teachers:teachers[identity]=forward(weights,'teacher_',x)
        y=teachers[identity];p=forward(weights,'student_',x);mse=float(weight@((p-y)**2))
        error=abs(mse-float(row['exact_gaussian_mse']))/max(mse,1e-14);assert error<1e-8;errors.append(error)
        if row['step']=='600':
            old=np.load(src/f'{row["teacher"]}_{row["seed"]}_{row["n"]}_{row["student"]}.npz')
            for key in weights.files:np.testing.assert_array_equal(weights[key],old[key])
        teacher={k.removeprefix('teacher_'):v for k,v in weights.items() if k.startswith('teacher_')}
        student={k.removeprefix('student_'):v for k,v in weights.items() if k.startswith('student_')}
        t=L.T@(feature_maps(teacher,8,keys)['oracle_top']@teacher['w'])
        maps=feature_maps(student,8,keys)
        for name,K in maps.items():
            W=L.T@K;beta=np.linalg.lstsq(W,t,rcond=1e-10)[0];residual=W@beta-t
            floor=float(residual@residual);error=abs(floor-float(row[name+'_population_floor']));assert error<1e-10;floors.append(error)
            assert mse>=floor-1e-9
    result=dict(status='passed',rows=180,exact_600_step_replays=60,max_relative_quadrature_error=max(errors),
                max_absolute_floor_error=max(floors),evidence_sha256=hashes,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                results_sha256=hashlib.sha256((out/'results.csv').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='evidence_sha256'})


if __name__=='__main__':main()
