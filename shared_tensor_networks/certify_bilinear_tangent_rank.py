"""Exact modular Jacobian minors for the archived dyadic bilinear initializations."""
import math
from pathlib import Path
import numpy as np
from run_svhn_standard import digest,write

PRIME=2147483647
assert all(PRIME%d for d in range(2,math.isqrt(PRIME)+1))


def residue(value):
    numerator,denominator=float(value).as_integer_ratio()
    assert denominator%PRIME
    return (numerator%PRIME)*pow(denominator,-1,PRIME)%PRIME


def forward(p,x):
    left=((x@p['A'].T)%PRIME)*((x@p['B'].T)%PRIME)%PRIME
    right=((x@p['AR'].T)%PRIME)*((x@p['BR'].T)%PRIME)%PRIME if 'AR' in p else left
    return (((left@p['C'].T)%PRIME)*((right@p['D'].T)%PRIME)%PRIME@p['w'])%PRIME


def jacobian(p,x):
    ax,bx=(x@p['A'].T)%PRIME,(x@p['B'].T)%PRIME;left=ax*bx%PRIME
    if 'AR' in p:ar,br=(x@p['AR'].T)%PRIME,(x@p['BR'].T)%PRIME;right=ar*br%PRIME
    else:right=left
    u,v=(left@p['C'].T)%PRIME,(right@p['D'].T)%PRIME;du,dv=v*p['w']%PRIME,u*p['w']%PRIME
    dl,dr=(du@p['C'])%PRIME,(dv@p['D'])%PRIME
    j=dict(C=du[:,:,None]*left[:,None,:]%PRIME,D=dv[:,:,None]*right[:,None,:]%PRIME,w=u*v%PRIME)
    if 'AR' in p:j.update(AR=(dr*br%PRIME)[:,:,None]*x[:,None,:]%PRIME,BR=(dr*ar%PRIME)[:,:,None]*x[:,None,:]%PRIME)
    else:dl=(dl+dr)%PRIME
    j.update(A=(dl*bx%PRIME)[:,:,None]*x[:,None,:]%PRIME,B=(dl*ax%PRIME)[:,:,None]*x[:,None,:]%PRIME)
    return np.concatenate([j[k].reshape(len(x),-1) for k in p],axis=1)


def pivots(matrix):
    a=[[int(v) for v in row] for row in matrix];row=0;columns=[]
    for col in range(len(a[0])):
        pivot=next((i for i in range(row,len(a)) if a[i][col]%PRIME),None)
        if pivot is None:continue
        a[row],a[pivot]=a[pivot],a[row];inverse=pow(a[row][col],-1,PRIME)
        a[row]=[(v*inverse)%PRIME for v in a[row]]
        for i in range(row+1,len(a)):
            multiplier=a[i][col];a[i]=[(v-multiplier*u)%PRIME for v,u in zip(a[i],a[row])]
        columns.append(col);row+=1
        if row==len(a):break
    return columns


def determinant(matrix):
    a=[[int(v) for v in row] for row in matrix];det=1
    for col in range(len(a)):
        pivot=next((i for i in range(col,len(a)) if a[i][col]%PRIME),None)
        if pivot is None:return 0
        if pivot!=col:a[col],a[pivot]=a[pivot],a[col];det=-det
        value=a[col][col];det=det*value%PRIME;inverse=pow(value,-1,PRIME)
        for i in range(col+1,len(a)):
            multiplier=a[i][col]*inverse%PRIME
            for j in range(col,len(a)):a[i][j]=(a[i][j]-multiplier*a[col][j])%PRIME
    return det%PRIME


def main():
    root=Path(__file__).parent;out=root/'results/bilinear_tangent_certificate';out.mkdir(exist_ok=True)
    x=np.array([(1,s,t) for s in range(5) for t in range(5-s)],dtype=object);rows=[]
    for seed in [211001,211002,211003]:
        for student in ['tied_raw','untied_raw']:
            source=root/f'results/bilinear_alternation_long/tied_{seed}_{student}_0.003_joint.npz';data=np.load(source)
            p={k[8:]:np.vectorize(residue,otypes=[object])(data[k]) for k in data.files if k.startswith('initial_')}
            J=jacobian(p,x);labels=[];independent=[]
            # Each individual parameter appears with degree<=2 in the raw output.
            # Central differences at h=1 and h=2 are therefore exact derivatives.
            for name,a in p.items():
                for index in np.ndindex(a.shape):
                    labels.append(dict(parameter=name,index=list(index)));original=a[index];derivatives=[]
                    for h in [1,2]:
                        a[index]=(original+h)%PRIME;plus=forward(p,x)
                        a[index]=(original-h)%PRIME;minus=forward(p,x)
                        a[index]=original;derivatives.append((plus-minus)*pow(2*h,-1,PRIME)%PRIME)
                    np.testing.assert_array_equal(*derivatives);independent.append(derivatives[0])
            np.testing.assert_array_equal(J,np.stack(independent,axis=1))
            columns=pivots(J);assert len(columns)==15;det=determinant(J[:,columns]);assert det!=0
            row=dict(seed=seed,student=student,prime=PRIME,rank_lower_bound=15,rank_upper_bound=15,
                selected_columns=columns,selected_parameters=[labels[c] for c in columns],determinant_mod_prime=det,
                jacobian_mod_prime=np.asarray(J,dtype=np.int64).tolist(),input_sha256=digest(source),
                dimensions=dict(parameters=J.shape[1],evaluations=J.shape[0]))
            write(out/f'{seed}_{student}.json',row);rows.append({k:v for k,v in row.items() if k!='jacobian_mod_prime'})
    write(out/'certificate.json',dict(status='certified',probe_points=np.asarray(x,dtype=int).tolist(),rows=rows,
        independent_check='exact quadratic central difference, h=1 and2, every parameter and probe',
        normalization='raw polynomial; dividing by a fixed nonzero student scale preserves rank',source_sha256=digest(Path(__file__))))
    print([(r['seed'],r['student'],r['determinant_mod_prime']) for r in rows])


if __name__=='__main__':main()
