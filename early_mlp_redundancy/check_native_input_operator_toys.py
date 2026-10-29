"""M77 fixed exact-function and implementation-distinction toy preflight."""
import json
import numpy as np
import torch
from native_input_operator import write,jvp,predictions
from fit_pca_encoder import sha,V

def main():
    torch.set_num_threads(1);rng=np.random.default_rng(7701);rows=[]
    # Cases1/2: z0=1 affine slice yields z1, versus genuine z1*z2 dependence.
    for function in ['linear_slice','product']:
        L=torch.tensor([[1.,0.,0.]] if function=='linear_slice' else [[0.,1.,0.]],dtype=torch.float64)
        R=torch.tensor([[0.,1.,0.]] if function=='linear_slice' else [[0.,0.,1.]],dtype=torch.float64)
        D=torch.ones(1,1,dtype=torch.float64);rec=torch.from_numpy(rng.normal(size=(16,3)));don=torch.from_numpy(rng.normal(size=(16,3)));rec[:,0]=1;don[:,0]=1
        if function=='product':rec[:,1]=1;rec[:,2]=0;don[:,1]=2;don[:,2]=2
        mean=torch.tensor([1.,0.,0.],dtype=torch.float64);target=write(L,R,D,don)-write(L,R,D,rec);pred=predictions(L,R,D,mean,rec,don)
        errors={k:float((x-target).square().sum()) for k,x in pred.items()}
        assert errors['midpoint_J_exact_control']<1e-25
        if function=='linear_slice':assert max(errors.values())<1e-25
        else:assert errors['shared_J_at_training_mean']>errors['recipient_J']>0
        implementations=[]
        # Same function: one channel; four equal positive copies; two large opposing copies.
        for name,coeff in [('single',[1.]),('distributed',[.25]*4),('cancelling',[2.,-1.])]:
            ll=L.repeat(len(coeff),1);rr=R.repeat(len(coeff),1);dd=torch.tensor([coeff],dtype=torch.float64)
            values=write(ll,rr,dd,don);derivative=jvp(ll,rr,dd,rec,don-rec)
            assert torch.allclose(values,write(L,R,D,don),atol=1e-12,rtol=1e-12)
            assert torch.allclose(derivative,jvp(L,R,D,rec,don-rec),atol=1e-12,rtol=1e-12)
            dropped=dd.clone();dropped[:,0]=0;deletion=float((write(ll,rr,dropped,don)-values).square().sum())
            implementations.append(dict(name=name,channels=len(coeff),first_channel_deletion_squared_error=deletion))
        assert implementations[1]['first_channel_deletion_squared_error']<implementations[0]['first_channel_deletion_squared_error']<implementations[2]['first_channel_deletion_squared_error']
        rows.append(dict(function=function,errors=errors,implementations=implementations))
    checks=[]
    for case in range(12):
        L=rng.normal(size=(7,5));R=rng.normal(size=(7,5));D=rng.normal(size=(3,7));anchor=rng.normal(size=5);direction=rng.normal(size=5)
        explicit=D@((R@anchor)[:,None]*L+(L@anchor)[:,None]*R)
        t=[torch.from_numpy(x) for x in [L,R,D,anchor,direction]];got=jvp(*t).numpy();assert np.allclose(got,explicit@direction,rtol=1e-12,atol=1e-12)
        h=1e-5
        f=lambda z:D@((L@z)*(R@z))
        finite=(f(anchor+h*direction)-f(anchor-h*direction))/(2*h)
        err=float(np.linalg.norm(finite-got)/np.linalg.norm(got));assert err<1e-8
        a=torch.from_numpy(anchor-direction/2);b=torch.from_numpy(anchor+direction/2)
        assert torch.allclose(write(*t[:3],b)-write(*t[:3],a),jvp(*t),rtol=1e-12,atol=1e-12)
        checks.append(dict(case=case,finite_difference_relative_error=err))
    out=dict(status='passed',source_sha256=sha(__file__),operator_source_sha256=sha(V.parent/'native_input_operator.py'),seed=7701,structured_cases=rows,random_cases=checks,scope='Fixed algebra/implementation tests. Function-identical implementations differ in deletion response; neither Jacobian equivalence nor toy behavior identifies native neuron redundancy.')
    path=V/'native_input_operator_toys.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print('PASS',len(rows),len(checks),max(x['finite_difference_relative_error'] for x in checks))
if __name__=='__main__':main()
