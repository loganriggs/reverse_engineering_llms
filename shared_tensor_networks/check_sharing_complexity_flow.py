"""Exact moment/rank fixtures and actual bilinear-network tangent-kernel checks."""
import json,math
from fractions import Fraction
from pathlib import Path
import torch
from torch.func import functional_call
from vision_circuits import BilinearCircuit

def powers(d,k):
    if d==1:return [(k,)]
    return [(i,)+r for i in range(k+1) for r in powers(d-1,k-i)]
def moment(a):
    return math.prod(math.prod(range(1,v,2)) if v%2==0 else 0 for v in a)
def rank_exact(a):
    a=[[Fraction(x) for x in row] for row in a];r=0
    for j in range(len(a[0])):
        pivot=next((i for i in range(r,len(a)) if a[i][j]),None)
        if pivot is None:continue
        a[r],a[pivot]=a[pivot],a[r];v=a[r][j];a[r]=[x/v for x in a[r]]
        for i in range(r+1,len(a)):
            v=a[i][j]
            if v:a[i]=[x-v*y for x,y in zip(a[i],a[r])]
        r+=1
    return r

def main():
    torch.set_num_threads(1);records=[]
    for d,k in [(2,1),(2,2),(3,2),(3,4)]:
        basis=powers(d,k);gram=[[moment(tuple(x+y for x,y in zip(a,b))) for b in basis] for a in basis]
        rank=rank_exact(gram);assert rank==math.comb(d+k-1,k)
        for beta in powers(d,2*k):
            coeff=Fraction(math.factorial(2*k)*moment(beta),math.prod(math.factorial(b) for b in beta)*math.prod(range(1,2*k,2)))
            expected=0 if any(b%2 for b in beta) else math.factorial(k)//math.prod(math.factorial(b//2) for b in beta)
            assert coeff==expected
        records.append(dict(d=d,k=k,exact_flattening_rank=rank,coefficient_identity=True))
    kernels=[]
    for depth in [2,3]:
        for norm in [False,True]:
            shared=BilinearCircuit(3,2,depth,2,False,norm,3,torch.float64);tree=BilinearCircuit(3,2,depth,2,True,norm,3,torch.float64)
            sn=list(shared.named_parameters());tn=list(tree.named_parameters());ranges={};offset=0
            for n,p in sn:ranges[n]=list(range(offset,offset+p.numel()));offset+=p.numel()
            mapping=[]
            for n,p in tn:
                target=n if n=='W' else n.split('.')[0]+'.'+str(tree.levels[int(n.split('.')[1])])
                mapping+=ranges[target]
            S=torch.zeros(len(mapping),offset,dtype=torch.float64);S[torch.arange(len(mapping)),mapping]=1
            theta=torch.cat([p.flatten() for _,p in sn]).detach();z=S@theta
            x=torch.randn(4,3,generator=torch.Generator().manual_seed(9),dtype=torch.float64)
            def output(model,named,v):
                params={};i=0
                for n,p in named:params[n]=v[i:i+p.numel()].reshape(p.shape);i+=p.numel()
                return functional_call(model,params,(x,)).flatten()
            J=torch.autograd.functional.jacobian(lambda v:output(tree,tn,v),z)
            Js=torch.autograd.functional.jacobian(lambda v:output(shared,sn,v),theta)
            torch.testing.assert_close(Js,J@S,rtol=1e-10,atol=1e-10)
            M=S.T@S;P=S@torch.linalg.solve(M,S.T);I=torch.eye(len(mapping),dtype=torch.float64)
            free=J@J.T;proj=J@P@J.T;tied=Js@Js.T;extra=J@(I-P)@J.T
            torch.testing.assert_close(free-proj,extra,rtol=1e-10,atol=1e-10)
            for positive in [extra,tied-M.diag().min()*proj,M.diag().max()*proj-tied]:
                assert torch.linalg.eigvalsh(positive).min()>-1e-9*max(1,float(positive.norm()))
            e=output(tree,tn,z)-torch.arange(8,dtype=torch.float64)/8
            disagreement=(I-P)@J.T@e
            torch.testing.assert_close(e@(free-proj)@e,disagreement.square().sum(),rtol=1e-10,atol=1e-10)
            kernels.append(dict(depth=depth,rmsnorm=norm,max_multiplicity=float(M.diag().max()),disagreement_gradient_energy=float(disagreement.square().sum()),status='passed'))
    out=Path(__file__).parent/'results/sharing_complexity_flow';out.mkdir(exist_ok=False)
    (out/'checks.json').write_text(json.dumps(dict(status='passed',exact_rank_fixtures=records,kernel_checks=kernels),indent=2)+'\n');print('Passed exact coefficient/rank checks and four actual-network kernel checks')
if __name__=='__main__':main()
