"""Sustained timing of proposed crossed grid; no scientific training outcomes."""
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from batched_coverage import BatchedCoverage,clip_each
from check_streamed_coverage import backward_streamed

def main():
 out=Path('results/h128_grid_preflight');out.mkdir(exist_ok=False)
 torch.set_num_threads(1);torch.manual_seed(265401);torch.backends.cuda.matmul.allow_tf32=False
 records=[]
 for n in [25000,100000]:
  for tasks in [2064,8256]:
   A=torch.randn(1,128,256,device='cuda')/16;B=torch.randn_like(A)/16;W=torch.randn(1,tasks,8256,device='cuda')/8256**.5
   model=BatchedCoverage(A,B,W);del A,B,W
   x=torch.randn(n,256,device='cuda');y=torch.randn(n,tasks,device='cuda');opt=torch.optim.Adam(model.parameters(),lr=.003)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
   for step in range(12):
    begin=time.perf_counter();opt.zero_grad(set_to_none=True);loss=backward_streamed(model,x,y,512);norm=clip_each(model)
    assert torch.isfinite(loss).all() and torch.isfinite(norm).all();opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-begin)
   r=dict(n=n,tasks=tasks,steps=12,warmup_steps=2,seconds_per_step=times,steady_median_seconds=float(np.median(times[2:])),peak_allocated_bytes=torch.cuda.max_memory_allocated())
   records.append(r);(out/'progress.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(r),flush=True)
   del model,x,y,opt,loss,norm;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
 estimate=sum(r['steady_median_seconds']*3000*15 for r in records)/3600
 result=dict(scope='Synthetic random targets; resource timing, not learning or guaranteed long-run throughput',records=records,estimated_60_fit_training_hours=estimate,estimate_excludes='target generation, normalization, checkpoints, analysis and contention; 15 fits per condition',sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['benchmark_h128_grid.py','check_streamed_coverage.py','active_coverage.py','batched_coverage.py']})
 (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print('Estimated training hours',estimate,flush=True)
if __name__=='__main__':main()
