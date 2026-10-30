"""Check exact full-batch gradient accumulation; no minibatch optimizer steps."""
import copy,gc,hashlib,json,time
from pathlib import Path
import torch
from active_coverage import active_losses
from batched_coverage import BatchedCoverage,clip_each

def backward_streamed(model,x,y,block_size):
    total=torch.zeros(len(model.A),device=x.device,dtype=x.dtype)
    for start in range(0,len(x),block_size):
        end=min(start+block_size,len(x))
        loss=active_losses(model,x[start:end],y[start:end])*((end-start)/len(x))
        loss.sum().backward();total+=loss.detach()
    return total

def main():
    torch.set_num_threads(1);torch.manual_seed(265001)
    checks=[]
    for device,dtype,tol in [('cpu',torch.float64,1e-10),('cuda',torch.float32,2e-5)]:
        torch.backends.cuda.matmul.allow_tf32=False
        A=torch.randn(2,5,7,device=device,dtype=dtype)/7**.5
        B=torch.randn_like(A)/7**.5;W=torch.randn(2,18,15,device=device,dtype=dtype)/15**.5
        original=BatchedCoverage(A,B,W);streamed=copy.deepcopy(original)
        x=torch.randn(73,7,device=device,dtype=dtype);y=torch.randn(73,11,device=device,dtype=dtype)
        opts=[torch.optim.Adam(m.parameters(),lr=.003) for m in [original,streamed]]
        max_grad=max_param=0.
        for step in range(5):
            for opt in opts:opt.zero_grad()
            full=active_losses(original,x,y);full.sum().backward()
            chunk=backward_streamed(streamed,x,y,16)
            torch.testing.assert_close(full,chunk,rtol=tol,atol=tol)
            for p,q in zip(original.parameters(),streamed.parameters()):
                torch.testing.assert_close(p.grad,q.grad,rtol=tol,atol=tol)
                max_grad=max(max_grad,float((p.grad-q.grad).abs().max()))
            for m,opt in zip([original,streamed],opts):clip_each(m);opt.step()
            for p,q in zip(original.parameters(),streamed.parameters()):
                torch.testing.assert_close(p,q,rtol=tol,atol=tol)
                max_param=max(max_param,float((p-q).detach().abs().max()))
        torch.testing.assert_close(streamed.W[:,11:],W[:,11:],rtol=0,atol=0)
        checks.append(dict(device=device,steps=5,rows=73,chunk=16,max_gradient_difference=max_grad,max_parameter_difference=max_param,inactive_unchanged=True))
    del original,streamed,opts,opt,m,p,q,x,y,A,B,W,full,chunk
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    out=Path('results/streamed_coverage_check_retry');out.mkdir(exist_ok=False)
    result=dict(status='passed',checks=checks,scope='Accumulation with one clip and Adam step per complete dataset, including uneven final chunk and inactive heads',sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['check_streamed_coverage.py','active_coverage.py','batched_coverage.py']})
    (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
