"""Prespecified larger pilot and independent evaluation, depths4/5."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from directional_sampling import DirectionalMixture,fit_axes
from deep_multitask import DeepMultiTask
from homogeneous_evaluation import sphere_from_gaussian


def main():
    torch.set_num_threads(1)
    root=Path(__file__).parent;out=root/'results/directional_refinement';out.mkdir(exist_ok=False)
    protocol=dict(depths=[4,5],batches=12,n=32768,pilot_directions=64,steps=200,pilot_seed=112000,evaluation_seed=113000,
                  stretches=[32,128],uniform_fraction=.2,calibration_seed=101000,
                  endpoint='teacher energy and INITIAL student error; same teachers as development',
                  gate='both quantities batch CV below .1; descriptive precision gate, not proof of accuracy or coverage',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    cal=torch.from_numpy(sphere_from_gaussian(np.random.default_rng(101000).normal(size=(4096,16))))
    rng=np.random.default_rng(113000);results=[]
    for depth in [4,5]:
        teacher=DeepMultiTask(16,8,6,depth,True,102000+depth);teacher.calibrate(cal)
        student=DeepMultiTask(16,8,6,depth,True,103000);student.calibrate(cal)
        pilot=torch.from_numpy(np.random.default_rng(112000+depth).normal(size=(64,16)))
        axes=fit_axes(teacher,student,pilot,steps=200);proposal=DirectionalMixture(axes,stretches=(32,128))
        np.save(out/f'axes_depth{depth}.npy',axes);rows=[]
        for batch in range(12):
            # Chunking bounds proposal-density memory (32768 x128 otherwise).
            weighted=[];weights=[]
            for chunk in range(8):
                u,w=proposal.sample(4096,rng)
                with torch.no_grad():
                    x=torch.from_numpy(4*u);y=teacher(x);pred=student(x)
                    a=np.column_stack([y.square().mean(dim=1).numpy(),(pred-y).square().mean(dim=1).numpy()])*w[:,None]
                weighted.append(a);weights.append(w)
            a=np.concatenate(weighted);w=np.concatenate(weights)
            np.savez_compressed(out/f'depth{depth}_batch{batch}.npz',weighted_values=a,weights=w)
            rows.append(dict(batch=batch,means=a.mean(axis=0).tolist(),ess=(a.sum(axis=0)**2/(a*a).sum(axis=0)).tolist(),weight_mean=float(w.mean())))
        means=np.array([r['means'] for r in rows]);cv=means.std(axis=0,ddof=1)/means.mean(axis=0)
        result=dict(depth=depth,rows=rows,mean=means.mean(axis=0).tolist(),batch_cv=cv.tolist(),precision_gate=bool(np.all(cv<.1)))
        results.append(result);print(result['depth'],result['mean'],'CV',result['batch_cv'],'gate',result['precision_gate'],flush=True)
    (out/'results.json').write_text(json.dumps(dict(status='complete',results=results),indent=2)+'\n')


if __name__=='__main__':main()
