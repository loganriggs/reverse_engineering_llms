"""Train and persist groups; population analysis is intentionally separate."""
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from batched_coverage import BatchedCoverage,clip_each
from active_coverage import active_losses
from quartic_cross_gram import product_factors,cross_gram
from regenerated_coverage import blocks,digest,save_readout_chunks

ROOT=Path('results/task_coverage_h64');D,H,P=128,64,2080
def sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def normalize(A,B,W):
    G=cross_gram(product_factors(A,B,np.ones(H)),product_factors(A,B,np.ones(H)))
    scale=np.sqrt(np.sum((W@G)*W,axis=1));assert np.isfinite(scale).all() and min(scale)>0
    return W/scale[:,None]

def main(smoke=False):
 global ROOT
 if smoke:ROOT=Path('/tmp/task_coverage_h64_smoke')
 steps=2 if smoke else 3000
 ROOT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 sources=['train_task_coverage_h64.py','active_coverage.py','batched_coverage.py','regenerated_coverage.py','quartic_cross_gram.py','TASK_COVERAGE_H64_PLAN.md']
 write_json(ROOT/'protocol.json',dict(smoke=smoke,steps=steps,numpy=np.__version__,torch=torch.__version__,block_size=256,noise_sd=.05,device=torch.cuda.get_device_name(),sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources}))
 initials={};initial_dir=ROOT/'initials';initial_dir.mkdir();groups=[]
 for seed in [257001,257002]:
  print('Preparing initial dictionary',seed,flush=True);rng=np.random.default_rng(seed)
  A=rng.normal(size=(H,D))/np.sqrt(D);B=rng.normal(size=(H,D))/np.sqrt(D);W=normalize(A,B,rng.normal(size=(P,P))/np.sqrt(P))
  initials[str(seed)]=(A.astype(np.float32),B.astype(np.float32),W.astype(np.float32))
  np.savez_compressed(initial_dir/f'{seed}.npz',A=initials[str(seed)][0],B=initials[str(seed)][1],W=initials[str(seed)][2])
 del A,B,W
 for seed in range(256101,256102 if smoke else 256106):
  print('Preparing teacher',seed,flush=True);directory=ROOT/str(seed);directory.mkdir();rng=np.random.default_rng(seed)
  A=rng.normal(size=(H,D))/np.sqrt(D);B=rng.normal(size=(H,D))/np.sqrt(D);W=normalize(A,B,rng.normal(size=(P+4,P))/np.sqrt(P))
  assert np.linalg.matrix_rank(np.r_[A,B])==D and np.linalg.matrix_rank(W[:P])==P
  np.savez_compressed(directory/'teacher_factors.npz',A=A,B=B)
  chunks=save_readout_chunks(directory,'teacher_W',W)
  write_json(directory/'teacher.json',dict(seed=seed,readout_chunks=chunks,source_heads=P,heldout_heads=4,readout_sha256=sha(W)))
  xx=np.empty((32768,D));yy=np.empty((32768,P));block_records=[]
  for block in blocks(A,B,W[:P],seed+5000,32768):
   start=block['index']*256;xx[start:start+len(block['x'])]=block['x'];yy[start:start+len(block['x'])]=block['y']
   block_records.append(dict(index=block['index'],rows=len(block['x']),sha256=digest(block)))
  write_json(directory/'source_recipe.json',dict(seed=seed+5000,stream=0,n=32768,tasks=P,block_size=256,noise_sd=.05,numpy=np.__version__,blocks=block_records))
  adaptation=list(blocks(A,B,W[P:],seed+5000,8192,stream=1))
  np.savez_compressed(directory/'adaptation.npz',**{k:np.concatenate([b[k] for b in adaptation]) for k in ['x','y','noise']});del adaptation
  starts={**initials,'teacher':(A.astype(np.float32),B.astype(np.float32),W[:P].astype(np.float32))}
  for n in [8192,32768]:
   for tasks in [520,2080]:
    name=f'n{n}_t{tasks}';dest=directory/name;dest.mkdir();order=['257001','257002','teacher']
    aa,bb,ww=map(np.stack,zip(*(starts[k] for k in order)))
    model=BatchedCoverage(torch.tensor(aa,device='cuda'),torch.tensor(bb,device='cuda'),torch.tensor(ww,device='cuda'))
    x=torch.tensor(xx[:n],dtype=torch.float32,device='cuda');y=torch.tensor(yy[:n,:tasks],dtype=torch.float32,device='cuda')
    opt=torch.optim.Adam(model.parameters(),lr=.003);clipped=np.zeros(3,dtype=int);curves=[]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    for step in range(steps):
     opt.zero_grad();ls=active_losses(model,x,y);assert torch.isfinite(ls).all(),(seed,name,step,'loss')
     ls.sum().backward();gn=clip_each(model);assert torch.isfinite(gn).all(),(seed,name,step,'gradient');clipped+=(gn.detach().cpu().numpy()>10);opt.step()
     if (step+1)%200==0 or step+1==steps:
      curves.append(dict(step=step+1,preupdate_loss=ls.detach().cpu().tolist(),gradient_norm=gn.detach().cpu().tolist()));print(seed,name,'step',step+1,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
    resources=dict(seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    arrays=[p.detach().cpu().numpy().copy() for p in [model.A,model.B,model.W]];saved=[]
    for index,init in enumerate(order):
     np.testing.assert_array_equal(arrays[2][index,tasks:],ww[index,tasks:])
     file=dest/f'{init}.npz';np.savez_compressed(file,A=arrays[0][index],B=arrays[1][index],W=arrays[2][index]);assert file.stat().st_size<20*1024**2
     saved.append(dict(initialization=init,file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),initial_sha256={key:sha(v[index]) for key,v in [('A',aa),('B',bb),('W',ww)]},clipped_steps=int(clipped[index])))
    write_json(dest/'curves.json',curves)
    record=dict(teacher=seed,n=n,tasks=tasks,steps=steps,status='trained',active_parameters=2*H*D+tasks*P,resources=resources,cases=saved)
    write_json(dest/'trained.json',record);groups.append(record);write_json(ROOT/'training_progress.json',dict(groups=groups,completed_fits=3*len(groups)))
    print('Saved',seed,name,'seconds',round(elapsed,2),flush=True)
    del model,opt,x,y,ls,gn,arrays,aa,bb,ww;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
  del xx,yy,A,B,W,starts;gc.collect()
 write_json(ROOT/'training_complete.json',dict(status='trained_not_evaluated',groups=len(groups),fits=3*len(groups)))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');main(parser.parse_args().smoke)
