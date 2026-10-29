"""Equal-width reduction, NumPy replay, parameter counts and finite differences."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from vision_circuits import BilinearCircuit
from nonuniform_tree import NonuniformTree,count,budget_widths
from audit_fashion_pilot import forward

def main():
    torch.set_num_threads(1);rows=[]
    for depth in [2,3,4]:
        for norm in [False,True]:
            old=BilinearCircuit(5,3,depth,2,True,norm,7,torch.float64);new=NonuniformTree(5,[3]*depth,2,norm,7,torch.float64);x=torch.randn(6,5,generator=torch.Generator().manual_seed(8),dtype=torch.float64)
            torch.testing.assert_close(old(x),new(x),atol=0,rtol=0)
            for (n,a),(m,b) in zip(old.named_parameters(),new.named_parameters()):assert n==m;torch.testing.assert_close(a,b,atol=0,rtol=0)
            for model in [old,new]:model(x).square().mean().backward()
            for a,b in zip(old.parameters(),new.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
        widths=budget_widths(3072,32,depth);model=NonuniformTree(3072,widths);assert sum(p.numel() for p in model.parameters())==count(3072,widths);assert len({p.data_ptr() for p in model.parameters()})==len(list(model.parameters()));rows.append(dict(depth=depth,widths=widths,parameters=count(3072,widths)))
    model=NonuniformTree(3,[2,3,4],2,True,7,torch.float64);x=torch.randn(2,3,generator=torch.Generator().manual_seed(10),dtype=torch.float64,requires_grad=True);names=[n for n,_ in model.named_parameters()];params=tuple(model.parameters());fn=lambda xx,*ps:functional_call(model,dict(zip(names,ps)),(xx,))
    assert torch.autograd.gradcheck(fn,(x,)+params,eps=1e-6,atol=1e-5,rtol=1e-4)
    state={n:p.detach().numpy() for n,p in model.named_parameters()};reference=forward(dict(depth=3,rmsnorm=True,architecture='nonuniform_tree'),state,x.detach().numpy());np.testing.assert_allclose(reference,model(x).detach().numpy(),rtol=1e-12,atol=1e-12)
    out=Path(__file__).parent/'results/nonuniform_correctness';out.mkdir(exist_ok=False);(out/'checks.json').write_text(json.dumps(dict(status='passed',equal_width_cases=6,finite_difference_gradcheck=True,numpy_replay=True,budgets=rows),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
