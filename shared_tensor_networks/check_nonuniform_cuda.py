"""Check newly introduced varying-width trees on the planned GPU backend."""
import copy,json
from pathlib import Path
import torch
from nonuniform_tree import NonuniformTree,budget_widths

def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rows=[]
    for depth in [2,3,4]:
        widths=budget_widths(3072,32,depth);cpu=NonuniformTree(3072,widths,seed=200100);gpu=copy.deepcopy(cpu).cuda();x=torch.randn(32,3072,generator=torch.Generator().manual_seed(200900));y=torch.arange(32)%10;a=cpu(x);b=gpu(x.cuda());torch.nn.functional.cross_entropy(a,y).backward();torch.nn.functional.cross_entropy(b,y.cuda()).backward();output=float((a.detach()-b.detach().cpu()).norm()/a.detach().norm());gradient=max(float((p.grad-q.grad.cpu()).norm()/p.grad.norm().clamp_min(1e-12)) for p,q in zip(cpu.parameters(),gpu.parameters()));assert max(output,gradient)<5e-4;rows.append(dict(depth=depth,widths=widths,relative_output_error=output,max_relative_gradient_error=gradient))
    out=Path(__file__).parent/'results/nonuniform_correctness/cuda.json';out.write_text(json.dumps(dict(status='passed',tf32=False,rows=rows),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
