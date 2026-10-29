import json
from pathlib import Path
import torch
from batched_coverage import BatchedCoverage,losses,clip_each
torch.manual_seed(241001);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
A=torch.randn(3,3,6,dtype=torch.float64);B=torch.randn_like(A);W=torch.randn(3,6,6,dtype=torch.float64);x=torch.randn(30,6,dtype=torch.float64);y=torch.randn(30,6,dtype=torch.float64);mask=torch.zeros(3,30,6,dtype=torch.float64);mask[0,:20,:3]=1;mask[1,:,:3]=1;mask[2]=1
batch=BatchedCoverage(A,B,W);single=[BatchedCoverage(A[i:i+1],B[i:i+1],W[i:i+1]) for i in range(3)];op=torch.optim.Adam(batch.parameters(),lr=.003);ops=[torch.optim.Adam(m.parameters(),lr=.003) for m in single]
for step in range(5):
 op.zero_grad();losses(batch,x,y,mask).sum().backward();clip_each(batch);op.step()
 for i,(m,o) in enumerate(zip(single,ops)):
  o.zero_grad();losses(m,x,y,mask[i:i+1]).sum().backward();clip_each(m);o.step()
 for p,*others in zip(batch.parameters(),*(m.parameters() for m in single)):
  torch.testing.assert_close(p,torch.cat(others),rtol=1e-12,atol=1e-12)
cpu=BatchedCoverage(A.float(),B.float(),W.float());gpu=BatchedCoverage(A.float().cuda(),B.float().cuda(),W.float().cuda())
cp=cpu(x.float());gp=gpu(x.float().cuda());forward=float(torch.linalg.vector_norm(cp-gp.cpu())/torch.linalg.vector_norm(cp));assert forward<1e-5
losses(cpu,x.float(),y.float(),mask.float()).sum().backward();losses(gpu,x.float().cuda(),y.float().cuda(),mask.float().cuda()).sum().backward()
grad=max(float(torch.linalg.vector_norm(p.grad-q.grad.cpu())/torch.linalg.vector_norm(p.grad)) for p,q in zip(cpu.parameters(),gpu.parameters()));assert grad<1e-5
out=Path('results/task_coverage_h8');out.mkdir(exist_ok=True);(out/'checks.json').write_text(json.dumps(dict(separate_adam_equivalence=True,steps=5,cpu_cuda_forward_relative=forward,cpu_cuda_gradient_relative=grad),indent=2)+'\n');print(forward,grad)
