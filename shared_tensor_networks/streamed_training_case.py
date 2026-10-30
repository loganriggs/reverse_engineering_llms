"""One deterministic full-batch training case with bounded activation memory."""
import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from batched_coverage import BatchedCoverage,clip_each
from check_streamed_coverage import backward_streamed
from chunked_checkpoint import save_checkpoint

def train_case(A,B,W,x,y,directory,steps=3000,device='cuda',chunk_size=512):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    if steps<1 or chunk_size<1 or len(x)!=len(y) or y.shape[1]>len(W):raise ValueError('invalid training dimensions')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    arrays=[np.asarray(v,dtype=np.float32) for v in [A,B,W]]
    model=BatchedCoverage(*(torch.tensor(v[None],device=device) for v in arrays))
    xx=torch.tensor(x,dtype=torch.float32,device=device);yy=torch.tensor(y,dtype=torch.float32,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=.003);curves=[];clipped=0
    if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True);loss=backward_streamed(model,xx,yy,chunk_size)
        if not torch.isfinite(loss).all():raise FloatingPointError(f'loss at step {step}')
        norm=clip_each(model)
        if not torch.isfinite(norm).all():raise FloatingPointError(f'gradient at step {step}')
        clipped+=int(norm.item()>10);opt.step()
        if (step+1)%200==0 or step+1==steps:
            curves.append(dict(step=step+1,preupdate_loss=loss.item(),gradient_norm=norm.item()))
            (directory/'curves.json').write_text(json.dumps(curves,indent=2)+'\n')
    if device=='cuda':torch.cuda.synchronize()
    resources=dict(seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    final=[v.detach().cpu().numpy()[0].copy() for v in [model.A,model.B,model.W]]
    np.testing.assert_array_equal(final[2][y.shape[1]:],arrays[2][y.shape[1]:])
    save_checkpoint(directory/'checkpoint',*final)
    record=dict(status='trained_not_evaluated',steps=steps,n=len(x),tasks=y.shape[1],chunk_size=chunk_size,device=device,clipped_steps=clipped,resources=resources,active_parameters=2*A.size+y.shape[1]*W.shape[1],sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['streamed_training_case.py','check_streamed_coverage.py','batched_coverage.py','active_coverage.py','chunked_checkpoint.py']})
    (directory/'trained.json').write_text(json.dumps(record,indent=2)+'\n')
    del model,xx,yy,opt,loss,norm;gc.collect()
    if device=='cuda':torch.cuda.synchronize();torch.cuda.empty_cache()
    return record
