"""Verify every LS block optimum, frozen other rows, and exact final population loss."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root=Path(__file__).parent;out=root/'results/reuse_als';src=root/'results/reuse_sample_noise';assert json.loads((out/'completed.json').read_text())['fits']==54;protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256'];q=np.load(src/'evaluation.npz');qx=q['x'];qw=q['w'];blocks=0;records=[];worst=0.
    for path in sorted(out.glob('t*_g4.json')):
        r=json.loads(path.read_text());saved=np.load(path.with_suffix('.npz'));dpath=src/f"data_t{r['teacher_groups']}_s{r['seed']}.npz";assert digest(dpath)==r['data_sha256'];assert digest(src/path.name)==r['sgd_result_sha256'];assert digest(src/path.with_suffix('.npz').name)==r['initialization_sha256'];d=np.load(dpath);initial=np.load(src/path.with_suffix('.npz').name);np.testing.assert_array_equal(saved['states'][0],initial['initial']);assert len(saved['states'])==401
        n=r['ntrain'];x=d['x'][:n];target=d['clean'][:n]+r['noise_std']*d['noise'][:n];scale=float(saved['student_scale'])
        for step,(before,after) in enumerate(zip(saved['states'][:-1],saved['states'][1:])):
            i=step%4;np.testing.assert_array_equal(np.delete(before,i,axis=0),np.delete(after,i,axis=0));design=x*(np.prod(x@np.delete(before,i,axis=0).T,axis=1)/scale)[:,None];res=design@after[i]-target
            normalized=float(np.linalg.norm(design.T@res)/max(1,np.linalg.norm(design)*np.linalg.norm(target)));assert normalized<1e-10;worst=max(worst,normalized)
            value=float(np.mean(res**2));assert abs(value-saved['train_mse'][step+1])<1e-10*max(1,value);assert value<=saved['train_mse'][step]+1e-10*max(1,saved['train_mse'][step]);blocks+=1
        a=saved['states'][-1];teacher=np.prod((qx@d['teacher'].T)**(4//len(d['teacher'])),axis=1)/float(d['teacher_scale']);pred=np.prod(qx@a.T,axis=1)/scale;mse=float(qw@(pred-teacher)**2);assert abs(mse-r['population_mse'])<1e-9*max(1,mse)
        records.append(dict(name=r['name'],checkpoint_sha256=digest(path.with_suffix('.npz')),result_sha256=digest(path)))
    result=dict(status='passed',fits=len(records),verified_block_optima=blocks,max_normalized_normal_equation_residual=worst,records=records,source_sha256=digest(Path(__file__)));(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
