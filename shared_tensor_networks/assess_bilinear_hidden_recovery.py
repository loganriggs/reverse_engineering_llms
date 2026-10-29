"""Branchwise hidden-subspace recovery modulo basis changes and branch exchange."""
import itertools,json
from pathlib import Path
import numpy as np
from experiment import gram,multiply
from run_bilinear_gauss_newton import unpack
from run_svhn_standard import digest,write

KEYS=[(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
G=gram(KEYS);WHITEN=np.linalg.cholesky(G).T


def hidden_coefficients(p,right=False):
    a,b=(p['AR'],p['BR']) if right and 'AR' in p else (p['A'],p['B'])
    return np.array([a[:,0]*b[:,0],a[:,0]*b[:,1]+a[:,1]*b[:,0],a[:,0]*b[:,2]+a[:,2]*b[:,0],
        a[:,1]*b[:,1],a[:,1]*b[:,2]+a[:,2]*b[:,1],a[:,2]*b[:,2]])


def basis(matrix):
    u,s,v=np.linalg.svd(matrix,full_matrices=False);rank=int((s>s[0]*1e-12).sum());assert rank==3
    return u[:,:rank]


def output_coefficients(left,right,p,scale):
    u,v=left@p['C'].T,right@p['D'].T;out={}
    for column,weight in enumerate(p['w']):
        product=multiply(dict(zip(KEYS,u[:,column])),dict(zip(KEYS,v[:,column])))
        for key,value in product.items():out[key]=out.get(key,0)+weight*value/scale
    return out


def mse_coeff(a,b):
    keys=sorted(a.keys()|b.keys());v=np.array([a.get(k,0)-b.get(k,0) for k in keys]);return float(v@gram(keys)@v)


def main():
    root=Path(__file__).parent;out=root/'results/bilinear_hidden_recovery';out.mkdir(exist_ok=True)
    source=root/'results/bilinear_gauss_newton';audit=json.loads((source/'audit.json').read_text());assert audit['status']=='passed'
    z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)));x,w=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1)
    monomials=np.stack([np.prod(x**np.array(k),axis=1) for k in KEYS],axis=1);rows=[]
    # A change of basis cannot change subspace recovery, including signs/scales.
    fixture=np.random.default_rng(214001).normal(size=(6,3));transform=np.array([[2.,1.,0],[0,-3,1],[0,0,.5]])
    np.testing.assert_allclose(np.linalg.svd(basis(WHITEN@fixture).T@basis(WHITEN@fixture@transform),compute_uv=False),1,atol=1e-12)
    for file in sorted(source.glob('*_0.*.json')):
        record=json.loads(file.read_text());ref=next(r for r in audit['records'] if r['name']==record['name'])
        assert ref['artifact_sha256']==digest(file.with_suffix('.npz')) and ref['result_sha256']==digest(file)
        d=np.load(root/record['input_file']);initial={k[8:]:d[k] for k in d.files if k.startswith('initial_')}
        teacher={k[8:]:d[k] for k in d.files if k.startswith('teacher_') and k!='teacher_scale'}
        student=unpack(np.load(file.with_suffix('.npz'))['states'][-1],initial)
        tc=[hidden_coefficients(teacher,right) for right in [False,True]];sc=[hidden_coefficients(student,right) for right in [False,True]]
        # Choose branch exchange by invariant subspace overlap, never patch error.
        candidates=[]
        for order in [[0,1],[1,0]]:
            cosine=[np.linalg.svd(basis(WHITEN@tc[order[i]]).T@basis(WHITEN@sc[i]),compute_uv=False) for i in [0,1]]
            candidates.append((sum(float(c@c) for c in cosine),order,cosine))
        score,order,cosines=max(candidates,key=lambda value:value[0]);projected=[];maps=[];angles=[]
        for i in [0,1]:
            t,s=tc[order[i]],sc[i];a=np.sqrt(w)[:,None]*(monomials@t);b=np.sqrt(w)[:,None]*(monomials@s)
            quadrature_cosine=np.linalg.svd(basis(a).T@basis(b),compute_uv=False)
            np.testing.assert_allclose(cosines[i],quadrature_cosine,atol=1e-10,rtol=1e-10)
            mapping=np.linalg.lstsq(a,b,rcond=1e-12)[0]
            independent=np.linalg.solve(t.T@G@t,t.T@G@s)
            np.testing.assert_allclose(mapping,independent,atol=1e-9,rtol=1e-9)
            maps.append(mapping.tolist());projected.append(t@mapping)
            angles.append(float(np.degrees(np.arccos(np.clip(cosines[i].min(),0,1)))))
        scale=float(d['student_scale']);target=output_coefficients(*tc,teacher,float(d['teacher_scale']))
        original=output_coefficients(*sc,student,scale);original_mse=mse_coeff(original,target)
        assert abs(original_mse-record['curves'][-1]['mse'])<1e-9
        patches={}
        for name,left,right in [('left',projected[0],sc[1]),('right',sc[0],projected[1]),('both',*projected)]:
            coefficients=output_coefficients(left,right,student,scale)
            raw=(((monomials@left)@student['C'].T)*((monomials@right)@student['D'].T))@student['w']/scale
            original_values=(((monomials@sc[0])@student['C'].T)*((monomials@sc[1])@student['D'].T))@student['w']/scale
            difference=mse_coeff(coefficients,original);np.testing.assert_allclose(difference,w@(raw-original_values)**2,atol=1e-9,rtol=1e-9)
            patches[name]=dict(difference_from_original_mse=difference,target_mse=mse_coeff(coefficients,target))
        # End-to-end coordinate invariance of the replacement diagnostic.
        changed=dict(student)
        transforms=[np.array([[2.,1.,0],[0,-3.,1],[0,0,.5]]),np.array([[1.,0,1],[0,2.,0],[0,0,-1.]])]
        changed['C']=student['C']@np.linalg.inv(transforms[0]).T
        changed['D']=student['D']@np.linalg.inv(transforms[1]).T
        altered=output_coefficients(sc[0]@transforms[0],sc[1]@transforms[1],changed,scale)
        altered_patch=output_coefficients(projected[0]@transforms[0],projected[1]@transforms[1],changed,scale)
        assert mse_coeff(altered,original)<1e-20
        assert abs(mse_coeff(altered_patch,altered)-patches['both']['difference_from_original_mse'])<1e-9
        initial_sc=[hidden_coefficients(initial,right) for right in [False,True]]
        initial_candidates=[]
        for initial_order in [[0,1],[1,0]]:
            values=[np.linalg.svd(basis(WHITEN@tc[initial_order[i]]).T@basis(WHITEN@initial_sc[i]),compute_uv=False) for i in [0,1]]
            initial_candidates.append((sum(float(c@c) for c in values),values))
        _,initial_cosines=max(initial_candidates,key=lambda value:value[0])
        rows.append(dict(name=record['name'],teacher=record['teacher'],student=record['student'],seed=record['seed'],damping=record['damping'],
            population_mse=original_mse,near_exact_function=record['curves'][-1]['mse']<1e-12,
            teacher_branch_order=order,branch_principal_cosines=[c.tolist() for c in cosines],worst_angles_degrees=angles,
            initial_worst_angles_degrees=[float(np.degrees(np.arccos(np.clip(c.min(),0,1)))) for c in initial_cosines],
            projection_maps=maps,patches=patches,checkpoint_sha256=ref['artifact_sha256']))
    write(out/'results.json',dict(status='complete',rows=rows,checks='coefficient/quadrature subspaces and projections; independent patch losses; subspace and end-to-end patch GL invariance',
        source_sha256=digest(Path(__file__)),scope='teacher-informed evaluation only; no training supervision; branchwise linear equivalence is not every possible circuit equivalence'))
    for r in rows:
        if r['near_exact_function']:print(r['name'],r['worst_angles_degrees'],r['patches']['both'],flush=True)


if __name__=='__main__':main()
