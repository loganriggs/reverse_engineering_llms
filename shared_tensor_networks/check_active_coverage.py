import gc,json
from pathlib import Path
import torch
from batched_coverage import BatchedCoverage,losses,clip_each
from active_coverage import active_losses

torch.manual_seed(254001);torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=Path('results/active_coverage');out.mkdir(exist_ok=True)
conditions=[(20,3),(20,6),(30,3),(30,6)]
A=torch.randn(3,3,6,dtype=torch.float64);B=torch.randn_like(A);W=torch.randn(3,6,6,dtype=torch.float64)
x=torch.randn(30,6,dtype=torch.float64);y=torch.randn(30,6,dtype=torch.float64)
full=BatchedCoverage(A.repeat(4,1,1),B.repeat(4,1,1),W.repeat(4,1,1))
groups=[BatchedCoverage(A,B,W) for _ in conditions]
mask=torch.zeros(12,30,6,dtype=torch.float64)
for i,(n,t) in enumerate(conditions):mask[3*i:3*(i+1),:n,:t]=1
opt=torch.optim.Adam(full.parameters(),lr=.003);opts=[torch.optim.Adam(m.parameters(),lr=.003) for m in groups]
records=[]
for step in range(5):
 opt.zero_grad();ls=losses(full,x,y,mask);ls.sum().backward()
 for index,((n,t),m,o) in enumerate(zip(conditions,groups,opts)):
  o.zero_grad();part=active_losses(m,x[:n],y[:n,:t]);part.sum().backward()
  torch.testing.assert_close(part,ls[3*index:3*(index+1)],rtol=1e-12,atol=1e-12)
  for fp,p in zip(full.parameters(),m.parameters()):torch.testing.assert_close(p.grad,fp.grad[3*index:3*(index+1)],rtol=1e-12,atol=1e-12)
  clip_each(m);o.step();torch.testing.assert_close(m.W[:,t:],W[:,t:],rtol=0,atol=0)
 clip_each(full);opt.step()
 max_error=0.
 for fp,*parts in zip(full.parameters(),*(m.parameters() for m in groups)):
  combined=torch.cat(parts);torch.testing.assert_close(fp,combined,rtol=1e-12,atol=1e-12)
  max_error=max(max_error,float((fp-combined).abs().max().detach()))
 records.append(dict(step=step+1,max_parameter_error=max_error))
cpu=BatchedCoverage(A.float(),B.float(),W.float());gpu=BatchedCoverage(A.float().cuda(),B.float().cuda(),W.float().cuda())
cl=active_losses(cpu,x[:20].float(),y[:20,:3].float());gl=active_losses(gpu,x[:20].float().cuda(),y[:20,:3].float().cuda())
cl.sum().backward();gl.sum().backward()
error=max(float(torch.linalg.vector_norm(p.grad-q.grad.cpu())/torch.linalg.vector_norm(p.grad)) for p,q in zip(cpu.parameters(),gpu.parameters()));assert error<1e-5
result=dict(steps=records,conditions=conditions,models=12,group_size=3,inactive_weights_unchanged=True,cpu_cuda_gradient_relative=error)
(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
torch.cuda.synchronize();del gpu,gl;gc.collect();torch.cuda.empty_cache()
