"""Numerical tangent-space ranks and undamped linearized fits at paired starts."""
import itertools
from pathlib import Path
import numpy as np
from run_bilinear_gauss_newton import jacobian
from run_bilinear_alternation import evaluate
from run_svhn_standard import digest,write

root=Path(__file__).parent
z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)));x,w=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1)
rows=[]
for teacher in ['tied','untied']:
    for seed in [211001,211002,211003]:
        projectors=[]
        for student in ['tied_raw','untied_raw']:
            source=root/f'results/bilinear_alternation_long/{teacher}_{seed}_{student}_0.003_joint.npz';d=np.load(source)
            p={k[8:]:d[k] for k in d.files if k.startswith('initial_')};tp={k[8:]:d[k] for k in d.files if k.startswith('teacher_') and k!='teacher_scale'}
            scale=float(d['student_scale']);r=np.sqrt(w)*(evaluate(p,x)/scale-evaluate(tp,x)/float(d['teacher_scale']))
            J=np.sqrt(w)[:,None]*jacobian(p,x,scale);u,s,vh=np.linalg.svd(J,full_matrices=False);rank=int((s>s[0]*1e-10).sum());assert rank==15
            direction=-vh[:rank].T@((u[:,:rank].T@r)/s[:rank]);relative=float(np.linalg.norm(J@direction+r)/np.linalg.norm(r));assert relative<1e-8
            projectors.append(u[:,:rank]@u[:,:rank].T)
            rows.append(dict(teacher=teacher,seed=seed,student=student,rank=rank,smallest_nonzero_singular_ratio=float(s[14]/s[0]),
                relative_linearized_residual=relative,undamped_step_norm=float(np.linalg.norm(direction)),input_sha256=digest(source)))
        assert np.linalg.norm(projectors[0]-projectors[1])<1e-8
write(root/'results/bilinear_gauss_newton/tangents.json',dict(status='passed',rows=rows,
    scope='floating-point rank at relative threshold1e-10, not exact symbolic certificate',source_sha256=digest(Path(__file__))))
print('All12initial Jacobians numerically rank15; paired output tangent projectors match')
