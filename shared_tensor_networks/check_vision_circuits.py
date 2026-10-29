"""Occurrence multiplicity and RMSNorm gradient correctness."""
import json
from pathlib import Path
import torch
from vision_circuits import BilinearCircuit,parameter_count


def main():
    torch.set_num_threads(1);rows=[]
    for depth in [1,2,3,4]:
        for norm in [False,True]:
            s=BilinearCircuit(5,3,depth,2,False,norm,1,torch.float64)
            t=BilinearCircuit(5,3,depth,2,True,norm,1,torch.float64)
            assert len(t.A)==2**depth-1
            assert len({p.data_ptr() for p in t.A})==len(t.A)
            for m in [s,t]:assert sum(p.numel() for p in m.parameters())==parameter_count(5,3,depth,2,m.tree)
            x=torch.randn(7,5,generator=torch.Generator().manual_seed(2),dtype=torch.float64,requires_grad=True)
            y1,y2=s(x),t(x);torch.testing.assert_close(y1,y2,rtol=1e-11,atol=1e-11)
            g1=torch.autograd.grad(y1.square().mean(),[x]+list(s.parameters()))
            g2=torch.autograd.grad(y2.square().mean(),[x]+list(t.parameters()))
            torch.testing.assert_close(g1[0],g2[0],rtol=1e-10,atol=1e-10)
            # W is directly registered before parameter lists.
            sg=dict(zip([n for n,_ in s.named_parameters()],g1[1:]));tg=dict(zip([n for n,_ in t.named_parameters()],g2[1:]))
            torch.testing.assert_close(sg['W'],tg['W'],rtol=1e-10,atol=1e-10)
            for l in range(depth):
                for prefix in ['A','B']:
                    occurrence=sum(tg[f'{prefix}.{i}'] for i,level in enumerate(t.levels) if level==l)
                    torch.testing.assert_close(sg[f'{prefix}.{l}'],occurrence,rtol=1e-10,atol=1e-10)
            # An occurrence can change without changing any other occurrence.
            if depth>1:
                before=t.A[-2].detach().clone()
                with torch.no_grad():t.A[-1].add_(.01)
                torch.testing.assert_close(before,t.A[-2],rtol=0,atol=0)
                assert not torch.allclose(s(x),t(x))
            rows.append(dict(depth=depth,rmsnorm=norm,tree_nodes=len(t.A),status='passed'))
    root=Path(__file__).parent;out=root/'results/vision_correctness';out.mkdir(exist_ok=False)
    (out/'checks.json').write_text(json.dumps(dict(status='passed',rows=rows),indent=2)+'\n');print(json.dumps(rows))


if __name__=='__main__':main()
