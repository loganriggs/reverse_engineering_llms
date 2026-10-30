"""Resumable completed-case orchestration; scientific launch requires frozen plan."""
import argparse,fcntl,gc,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import lu_factor
from chunked_checkpoint import save_checkpoint,load_checkpoint
from quartic_cross_gram import product_factors,cross_gram
from regenerated_coverage import blocks,digest
from streamed_training_case import train_case

def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized(seed,d,h,heads):
    rng=np.random.default_rng(seed);A=rng.normal(size=(h,d))/np.sqrt(d);B=rng.normal(size=(h,d))/np.sqrt(d)
    F=product_factors(A,B,np.ones(h));G=cross_gram(F,F);W=rng.normal(size=(heads,len(F)))/np.sqrt(len(F))
    # Avoid an extra full W@G allocation: normalize task rows in fixed chunks.
    for start in range(0,heads,128):
        v=W[start:start+128];energy=np.sum((v@G)*v,axis=1);assert np.isfinite(energy).all() and min(energy)>0;v/=np.sqrt(energy[:,None])
    return A,B,W

def main(root,smoke=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    d,h=(4,2) if smoke else (256,128);p=h*(h+1)//2
    ns=[128,512] if smoke else [25000,100000];tasks=[1,3] if smoke else [2064,8256]
    teachers=[266101] if smoke else list(range(266101,266106));starts=[267001,267002];adapt=64 if smoke else 32768;steps=3 if smoke else 3000;device='cpu' if smoke else 'cuda'
    sources=['run_task_coverage_h128.py','streamed_training_case.py','check_streamed_coverage.py','chunked_checkpoint.py','regenerated_coverage.py','quartic_cross_gram.py','batched_coverage.py','active_coverage.py','TASK_COVERAGE_H128_PLAN.md']
    protocol=dict(smoke=smoke,d=d,h=h,p=p,ns=ns,tasks=tasks,teachers=teachers,starts=starts,adaptation=adapt,steps=steps,device=device,numpy=np.__version__,sources={n:sha(n) for n in sources})
    if not smoke:
        assert 'Status: FROZEN' in Path('TASK_COVERAGE_H128_PLAN.md').read_text(),'scientific protocol not frozen'
    if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol,'resume protocol differs'
    else:write(root/'protocol.json',protocol)
    for seed in starts:
        dest=root/'initials'/str(seed)
        if not dest.exists():save_checkpoint(dest,*[v.astype(np.float32) for v in normalized(seed,d,h,p)])
        else:load_checkpoint(dest)  # Verify the complete hashed artifact before reuse.
    completed=[]
    for seed in teachers:
        td=root/str(seed);td.mkdir(exist_ok=True);teacher=td/'teacher'
        if not teacher.exists():save_checkpoint(teacher,*normalized(seed,d,h,p+4))
        A,B,W=load_checkpoint(teacher)
        if not (td/'teacher_diagnostics.json').exists():
            lu,_=lu_factor(W[:p]);pivots=abs(np.diag(lu));assert np.isfinite(pivots).all() and min(pivots)>0
            factor_rank=int(np.linalg.matrix_rank(np.r_[A,B]));assert factor_rank==2*h
            write(td/'teacher_diagnostics.json',dict(factor_rank=factor_rank,min_abs_readout_lu_pivot=float(min(pivots)),scope='Floating LU nonsingularity diagnostic, not exact rank certificate'));del lu,pivots
        xx=np.empty((max(ns),d),dtype=np.float32);yy=np.empty((max(ns),p),dtype=np.float32);hashes=[]
        for block in blocks(A,B,W[:p],seed+5000,max(ns)):
            start=256*block['index'];end=start+len(block['x']);xx[start:end]=block['x'];yy[start:end]=block['y'];hashes.append(dict(index=block['index'],rows=len(block['x']),sha256=digest(block)))
        recipe=dict(seed=seed+5000,stream=0,n=max(ns),tasks=p,block_size=256,noise_sd=.05,blocks=hashes)
        if (td/'source_recipe.json').exists():assert json.loads((td/'source_recipe.json').read_text())==recipe
        else:write(td/'source_recipe.json',recipe)
        ad=list(blocks(A,B,W[p:],seed+5000,adapt,stream=1));adarrays={k:np.concatenate([b[k] for b in ad]) for k in ['x','y','noise']}
        if (td/'adaptation.npz').exists():
            with np.load(td/'adaptation.npz',allow_pickle=False) as old:
                for k,v in adarrays.items():np.testing.assert_array_equal(old[k],v)
        else:np.savez(td/'adaptation.npz',**adarrays)
        del ad,adarrays
        for n in ns:
            for t in tasks:
                for init in [str(v) for v in starts]+['teacher']:
                    dest=td/f'n{n}_t{t}'/init
                    if (dest/'trained.json').exists():
                        record=json.loads((dest/'trained.json').read_text());assert (record['n'],record['tasks'],record['steps'])==(n,t,steps);load_checkpoint(dest/'checkpoint')
                    else:
                        if dest.exists():raise RuntimeError(f'Incomplete case retained; explicit recovery required: {dest}')
                        if init=='teacher':ia,ib,iw=A.astype(np.float32),B.astype(np.float32),W[:p].astype(np.float32)
                        else:ia,ib,iw=load_checkpoint(root/'initials'/init)
                        record=train_case(ia,ib,iw,xx[:n],yy[:n,:t],dest,steps=steps,device=device,chunk_size=512)
                        del ia,ib,iw
                    completed.append(dict(teacher=seed,n=n,tasks=t,initialization=init,trained_sha256=sha(dest/'trained.json'),checkpoint_manifest_sha256=sha(dest/'checkpoint'/'checkpoint.json')))
                    write(root/'progress.json',dict(completed=completed));print('Completed',seed,n,t,init,flush=True)
        del xx,yy,A,B,W;gc.collect()
    write(root/'complete.json',dict(status='trained_not_evaluated',fits=len(completed),protocol_sha256=sha(root/'protocol.json')))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args();main(args.root,args.smoke)
