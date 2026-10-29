"""CPU/CUDA agreement before larger-data GPU work; no training sweep."""
import copy,json
from pathlib import Path
import torch
from vision_circuits import BilinearCircuit

def main():
    torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rows=[]
    for dtype in [torch.float64,torch.float32]:
        for depth in [2,4]:
            for tree in [False,True]:
                cpu=BilinearCircuit(3072,32,depth,10,tree,True,200100,dtype);gpu=copy.deepcopy(cpu).cuda();x=torch.randn(32,3072,generator=torch.Generator().manual_seed(200900),dtype=dtype);labels=torch.arange(32)%10
                a=cpu(x);b=gpu(x.cuda());la=torch.nn.functional.cross_entropy(a,labels);lb=torch.nn.functional.cross_entropy(b,labels.cuda());la.backward();lb.backward()
                tolerance=1e-9 if dtype==torch.float64 else 5e-4;output_error=float((a-b.cpu()).norm()/a.norm());gradient_error=max(float((p.grad-q.grad.cpu()).norm()/p.grad.norm().clamp_min(1e-12)) for p,q in zip(cpu.parameters(),gpu.parameters()));assert output_error<tolerance and gradient_error<tolerance
                rows.append(dict(dtype=str(dtype),depth=depth,tree=tree,relative_output_error=output_error,max_relative_gradient_error=gradient_error));del gpu;torch.cuda.empty_cache()
    out=Path(__file__).parent/'results/vision_cuda';out.mkdir(exist_ok=False);(out/'checks.json').write_text(json.dumps(dict(status='passed',device=torch.cuda.get_device_name(0),torch_version=torch.__version__,tf32=False,rows=rows),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
