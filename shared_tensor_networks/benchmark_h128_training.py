"""Real training-shape timing only; random targets are not scientific evidence."""
import gc,hashlib,json,time
from pathlib import Path
import torch
from batched_coverage import BatchedCoverage,clip_each
from check_streamed_coverage import backward_streamed

def main():
    torch.set_num_threads(1);torch.manual_seed(265101);torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    for tasks in [2064,8256]:
        A=torch.randn(1,128,256,device='cuda')/16;B=torch.randn_like(A)/16
        W=torch.randn(1,tasks,8256,device='cuda')/8256**.5
        model=BatchedCoverage(A,B,W);del A,B,W
        x=torch.randn(100000,256,device='cuda');y=torch.randn(100000,tasks,device='cuda')
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for step in range(2):
            begin=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=backward_streamed(model,x,y,512);norm=clip_each(model)
            assert torch.isfinite(loss).all() and torch.isfinite(norm).all();opt.step();torch.cuda.synchronize()
            times.append(time.perf_counter()-begin)
        record=dict(d=256,h=128,product_features=8256,tasks=tasks,n=100000,chunk=512,steps=2,seconds_per_step=times,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),parameters=sum(p.numel() for p in model.parameters()))
        records.append(record);print(json.dumps(record),flush=True)
        del model,opt,x,y,loss,norm;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    out=Path('results/h128_training_preflight');out.mkdir(exist_ok=False)
    result=dict(scope='Resource smoke only: random targets, two steps, no scientific normalization or convergence claim; analysis preflight still required',torch=torch.__version__,gpu=torch.cuda.get_device_name(),records=records,sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['benchmark_h128_training.py','check_streamed_coverage.py','active_coverage.py','batched_coverage.py']})
    (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
