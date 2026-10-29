"""Exact single-factor alternating least squares on archived finite-data tasks."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from audit_fixed_degree_reuse import coefficients,inner
from assess_fixed_degree_recovery import assess

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root=Path(__file__).parent;src=root/'results/reuse_sample_noise';out=root/'results/reuse_als';out.mkdir(exist_ok=True)
    protocol=dict(sweeps=100,factor_order=[0,1,2,3],solver='numpy lstsq rcond=None; no regularization or gauge changes',scope='same archived data, noisy targets, four-factor initialization and frozen normalization as SGD; 400block solves vs1000SGDsteps, not compute matched',source_sha256=digest(Path(__file__)),baseline_protocol_sha256=digest(src/'protocol.json'))
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    for path in sorted(src.glob('t*_g4.json')):
        name=path.stem;dest=out/f'{name}.json'
        if dest.exists():continue
        r=json.loads(path.read_text());dpath=src/f"data_t{r['teacher_groups']}_s{r['seed']}.npz";data=np.load(dpath);initpath=path.with_suffix('.npz');saved=np.load(initpath);a=saved['initial'].copy();scale=float(saved['student_scale']);n=r['ntrain'];x=data['x'][:n];target=data['clean'][:n]+r['noise_std']*data['noise'][:n];states=[a.copy()];losses=[float(np.mean((np.prod(x@a.T,axis=1)/scale-target)**2))];conditions=[];ranks=[];start=time.monotonic();failure=None
        for step in range(400):
            i=step%4;other=np.prod(x@np.delete(a,i,axis=0).T,axis=1)/scale;design=x*other[:,None];solution,_,rank,sv=np.linalg.lstsq(design,target,rcond=None)
            if not np.isfinite(solution).all():failure=dict(step=step,reason='nonfinite solution');break
            a[i]=solution;value=float(np.mean((np.prod(x@a.T,axis=1)/scale-target)**2));assert value<=losses[-1]+1e-10*max(1,losses[-1]);states.append(a.copy());losses.append(value);conditions.append(float(sv[0]/sv[-1]) if sv[-1]>0 else None);ranks.append(int(rank))
        tc=coefficients(data['teacher'],float(data['teacher_scale']));sc=coefficients(a,scale);delta={p:sc[p]-tc[p] for p in tc};pop=inner(delta,delta);recovery=assess(data['teacher'],a,float(data['teacher_scale']),scale)
        np.savez_compressed(out/f'{name}.npz',states=np.array(states),train_mse=np.array(losses),ranks=np.array(ranks),student_scale=scale)
        row={k:r[k] for k in ['teacher_groups','seed','ntrain','noise_std']};row.update(name=name,population_mse=pop,recovery=recovery,train_mse=losses[-1],max_block_condition=max(c for c in conditions if c is not None),failure=failure,seconds=time.monotonic()-start,data_sha256=digest(dpath),initialization_sha256=digest(initpath),sgd_result_sha256=digest(path));dest.write_text(json.dumps(row,indent=2)+'\n');print(name,pop,flush=True)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',fits=54))+'\n')
if __name__=='__main__':main()
