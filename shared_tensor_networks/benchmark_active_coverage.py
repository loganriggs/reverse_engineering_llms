import gc,json,time
from pathlib import Path
import torch
from batched_coverage import BatchedCoverage,clip_each
from active_coverage import active_losses

torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
root=Path('results/active_coverage');checks=json.loads((root/'checks.json').read_text());assert checks['inactive_weights_unchanged']
records=[]
for n,t in [(8192,520),(8192,2080),(32768,520),(32768,2080)]:
 torch.manual_seed(254002)
 x=torch.randn(n,128,device='cuda',dtype=torch.float32);y=torch.zeros(n,t,device='cuda',dtype=torch.float32)
 model=BatchedCoverage(torch.randn(3,64,128,device='cuda')/128**.5,torch.randn(3,64,128,device='cuda')/128**.5,torch.randn(3,2080,2080,device='cuda')/2080**.5)
 opt=torch.optim.Adam(model.parameters(),lr=.003);loss_values=[];status='complete';elapsed=None
 try:
  opt.zero_grad();ls=active_losses(model,x,y);assert torch.isfinite(ls).all();ls.sum().backward();gn=clip_each(model);assert torch.isfinite(gn).all();opt.step()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  for step in range(5):
   opt.zero_grad();ls=active_losses(model,x,y);assert torch.isfinite(ls).all();ls.sum().backward();gn=clip_each(model);assert torch.isfinite(gn).all();opt.step();loss_values.append(ls.detach().cpu().tolist())
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
 except torch.cuda.OutOfMemoryError:
  status='out_of_memory'
 row=dict(n=n,tasks=t,models=3,d=128,h=64,status=status,timed_steps=5 if status=='complete' else len(loss_values),seconds=elapsed,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),losses=loss_values)
 records.append(row);(root/'resources.json').write_text(json.dumps(records,indent=2)+'\n');print(row,flush=True)
 del model,opt,x,y
 if 'ls' in globals():del ls
 if 'gn' in globals():del gn
 gc.collect();torch.cuda.empty_cache()
