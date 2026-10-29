"""Frozen-data learning-rate and local-basin controls."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask
from directional_sampling import DirectionalMixture,fit_axes


def main():
    torch.set_num_threads(1);root=Path(__file__).parent;src=root/'results/coverage_training';out=root/'results/depth5_controls';out.mkdir(exist_ok=False)
    configurations=[('random_lr1e4',1e-4,False),('random_lr1e5',1e-5,False),('teacher_near',1e-4,True)]
    protocol=dict(depth=5,d=16,h=8,tasks=6,seeds=[0,1,2],steps=600,clip=10,configs=configurations,
                  teacher_near_noise='independent Gaussian, .01 times each parameter tensor RMS; fixed teacher buffers; seed130000+seed',
                  data='same frozen1024 importance-weighted examples from coverage_training; no new training labels',
                  evaluation='fresh pilot64axes200steps;6batches16384; pilot132000+10*seed+configindex; samples131000+10*seed+configindex',
                  limitations='local-basin positive control uses true teacher parameters; not learnability from data or sample-efficiency evidence',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('d5*') if p.suffix=='.pt' or p.name.endswith('_training.npz')})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());rows=[]
    for seed in range(3):
        saved=torch.load(src/f'd5_s{seed}_initial.pt',weights_only=True)
        teacher=DeepMultiTask(16,8,6,5,True,0);teacher.load_state_dict(saved['teacher'])
        data=np.load(src/f'd5_s{seed}_importance_training.npz');x=torch.from_numpy(data['inputs']);y=torch.from_numpy(data['targets']);w=torch.from_numpy(data['weights'])
        for ci,(name,lr,near) in enumerate(configurations):
            model=DeepMultiTask(16,8,6,5,True,0);model.load_state_dict(saved['teacher'] if near else saved['initial'])
            if near:
                g=torch.Generator().manual_seed(130000+seed)
                with torch.no_grad():
                    for p in model.parameters():p.add_(.01*p.square().mean().sqrt()*torch.randn(p.shape,generator=g,dtype=p.dtype))
            prefix=f's{seed}_{name}';torch.save(model.state_dict(),out/f'{prefix}_initial.pt');opt=torch.optim.Adam(model.parameters(),lr=lr);trace=[];clipped=0;failure=None
            def measure(step):
                opt.zero_grad();pred=model(x);loss=((pred-y).square().mean(dim=1)*w).mean();loss.backward()
                grad=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
                with torch.no_grad():
                    a=(w[:,None]*pred.square()).mean();b=(w[:,None]*y.square()).mean();c=(w[:,None]*pred*y).mean()
                trace.append(dict(step=step,training_loss=float(loss.detach()),gradient_norm=grad,prediction_energy=float(a),target_energy=float(b),alignment=float(c/torch.sqrt(a*b).clamp_min(1e-300)),parameter_norms={k:float(v.detach().norm()) for k,v in model.named_parameters()}))
            measure(0)
            for step in range(600):
                opt.zero_grad();loss=((model(x)-y).square().mean(dim=1)*w).mean()
                if not torch.isfinite(loss):failure='nonfinite loss';break
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(gn):failure='nonfinite gradient';break
                clipped+=float(gn)>10;opt.step()
                if step+1 in [1,10,50,100,300,600]:measure(step+1)
            torch.save(model.state_dict(),out/f'{prefix}.pt')
            pilot=torch.from_numpy(np.random.default_rng(132000+10*seed+ci).normal(size=(64,16)))
            axes=fit_axes(teacher,model,pilot,steps=200);q=DirectionalMixture(axes,(32,128));rng=np.random.default_rng(131000+10*seed+ci);batches=[];allvalues=[]
            for batch in range(6):
                values=[]
                for chunk in range(4):
                    u,ww=q.sample(4096,rng)
                    with torch.no_grad():
                        xx=torch.from_numpy(4*u);yy=teacher(xx).numpy();pp=model(xx).numpy()
                    v=np.column_stack([(yy*yy).mean(axis=1),((pp-yy)**2).mean(axis=1),(pp*pp).mean(axis=1),(pp*yy).mean(axis=1)])*ww[:,None];values.append(v)
                values=np.concatenate(values);batches.append(values.mean(axis=0));allvalues.append(values)
            batches=np.array(batches);mean=batches.mean(axis=0);cv=batches[:,:2].std(axis=0,ddof=1)/mean[:2]
            np.savez_compressed(out/f'{prefix}_evaluation.npz',axes=axes,weighted_values=np.concatenate(allvalues))
            row=dict(seed=seed,config=name,config_index=ci,lr=lr,relative_mse=float(mean[1]/mean[0]),prediction_to_target_energy=float(mean[2]/mean[0]),alignment=float(mean[3]/np.sqrt(mean[0]*mean[2])),batch_cv=cv.tolist(),precision_gate=bool(np.all(cv<.1)),batches=batches.tolist(),trace=trace,clipped_steps=clipped,failure=failure,completed_steps=step+1)
            rows.append(row);(out/'results.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n');print(seed,name,row['relative_mse'],row['alignment'],'gate',row['precision_gate'],flush=True)
    (out/'results.json').write_text(json.dumps(dict(status='complete',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
