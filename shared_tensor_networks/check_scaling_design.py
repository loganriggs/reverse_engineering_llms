"""Budget, seed separation, source isolation and frozen-source checks."""
import hashlib,json
from pathlib import Path
import torch
from deep_multitask import DeepMultiTask
from run_scaling_grid import count,budget_width


def main():
    root=Path(__file__).parent;out=root/'results/scaling_grid';p=json.loads((out/'protocol.json').read_text())
    assert hashlib.sha256((root/'run_scaling_grid.py').read_bytes()).hexdigest()==p['source_sha256']
    assert (out/'source.py').read_bytes()==(root/'run_scaling_grid.py').read_bytes()
    for name,digest in p['dependencies'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    seeds=set();budgets=[];checks=[]
    for gi,(d,h,depth) in enumerate(p['grid']):
        wide=budget_width(d,h,depth)
        for width,shared in [(h,True),(h,False),(wide,True)]:
            model=DeepMultiTask(d,width,12,depth,shared,0)
            assert sum(v.numel() for v in model.parameters())==count(d,width,depth,shared)
        budgets.append(dict(d=d,h=h,depth=depth,budget_ratio=count(d,wide,depth,True)/count(d,h,depth,False)))
        for ki in range(2):
            for rep in range(3):
                base=2000000+10000000*gi+1000000*ki+10000*rep
                used=[base] if ki==0 else [base+100*(t+1) for t in range(16)]
                assert not seeds.intersection(used);seeds.update(used)
        for shared in [True,False]:
            teacher=DeepMultiTask(d,h,16,depth,shared,145000+gi)
            x=torch.randn(17,d,generator=torch.Generator().manual_seed(145100+gi),dtype=torch.float64,requires_grad=True)
            before=teacher(x)[:,:12];grad,=torch.autograd.grad(before.sum(),x)
            with torch.no_grad():
                teacher.W[12:].mul_(19)
                if not shared:
                    for a,b in zip(teacher.A,teacher.B):a[12:].add_(.3);b[12:].mul_(2)
            after=teacher(x)[:,:12];grad2,=torch.autograd.grad(after.sum(),x)
            torch.testing.assert_close(before,after,rtol=0,atol=0);torch.testing.assert_close(grad,grad2,rtol=0,atol=0)
            checks.append(dict(grid_index=gi,shared=shared,heldout_mutation_source_invariant=True))
    result=dict(status='passed',distinct_teacher_initialization_seeds=len(seeds),budgets=budgets,source_isolation_checks=checks,
                scope='source function/input-gradient isolation and seed/budget/source integrity; final audit separately checks actual data/results')
    (out/'design_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
