"""Paired rare-direction coverage diagnostic, not a label-budget comparison."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask
from directional_sampling import DirectionalMixture,fit_axes
from homogeneous_evaluation import sphere_from_gaussian


def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/coverage_training';out.mkdir(exist_ok=False)
    protocol=dict(d=16,h=8,tasks=6,depths=[3,5],teacher_seeds=[120000,120001,120002],student_seeds=[121000,121001,121002],
                  steps=600,ntrain=1024,lr=.001,clip=10,calibration_samples=32768,evaluation_batches=6,evaluation_batch_size=8192,
                  pilot_axes=64,pilot_steps=200,stretches=[32,128],uniform_fraction=.2,
                  calibration_rng='122000+100*depth+seed',training_rng='123000+100*depth+seed',evaluation_rng='124000+100*depth+seed',
                  pilot_rng='125000+100*depth+seed; evaluation pilot126000+100*depth+10*seed+method_index',
                  comparison='fixed uniform vs exact importance-weighted sphere training; same initialized shared student; same1024 training labels',
                  limitation='teacher-gradient access and32768 calibration labels are additional privileged supervision; not an equal-total-label sample-efficiency claim; no task holdout in this diagnostic',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());rows=[]
    for depth in [3,5]:
        for seed in range(3):
            calrng=np.random.default_rng(122000+100*depth+seed)
            cal=torch.from_numpy(sphere_from_gaussian(calrng.normal(size=(4096,16))))
            teacher=DeepMultiTask(16,8,6,depth,True,120000+seed);teacher.calibrate(cal)
            initial=DeepMultiTask(16,8,6,depth,True,121000+seed);initial.calibrate(cal)
            pilot=torch.from_numpy(np.random.default_rng(125000+100*depth+seed).normal(size=(64,16)))
            axes=fit_axes(teacher,initial,pilot,steps=200);proposal=DirectionalMixture(axes,(32,128))
            energies=[]
            for _ in range(8):
                u,w=proposal.sample(4096,calrng);x=torch.from_numpy(u*4)
                with torch.no_grad():energies.append(np.stack([(teacher(x).numpy()**2*w[:,None]).mean(axis=0),(initial(x).numpy()**2*w[:,None]).mean(axis=0)]))
            energies=np.array(energies);scales=np.sqrt(energies.mean(axis=0))
            with torch.no_grad():teacher.W.div_(torch.from_numpy(scales[0,:,None]));initial.W.div_(torch.from_numpy(scales[1,:,None]))
            base=f'd{depth}_s{seed}';np.savez_compressed(out/f'{base}_calibration.npz',axes=axes,batch_energies=energies,scales=scales)
            torch.save(dict(teacher=teacher.state_dict(),initial=initial.state_dict()),out/f'{base}_initial.pt')
            trainrng=np.random.default_rng(123000+100*depth+seed)
            for method_index,method in enumerate(['uniform','importance']):
                if method=='uniform':u=sphere_from_gaussian(trainrng.normal(size=(1024,16)))/4;w=np.ones(1024)
                else:u,w=proposal.sample(1024,trainrng)
                x=torch.from_numpy(4*u);weights=torch.from_numpy(w)
                with torch.no_grad():y=teacher(x)
                model=copy.deepcopy(initial);opt=torch.optim.Adam(model.parameters(),lr=.001);start=time.monotonic();clipped=0;failure=None;curve=[]
                for step in range(600):
                    opt.zero_grad();loss=((model(x)-y).square().mean(dim=1)*weights).mean()
                    if not torch.isfinite(loss):failure='nonfinite loss';break
                    loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                    if not torch.isfinite(gn):failure='nonfinite gradient';break
                    clipped+=float(gn)>10;opt.step()
                    if (step+1)%100==0:curve.append(dict(step=step+1,pre_update_loss=float(loss.detach()),gradient_norm=float(gn)))
                prefix=f'{base}_{method}';torch.save(model.state_dict(),out/f'{prefix}.pt')
                np.savez_compressed(out/f'{prefix}_training.npz',inputs=x.numpy(),weights=w,targets=y.numpy())
                # A fresh fixed proposal for this final model; no training changes follow.
                pilot=torch.from_numpy(np.random.default_rng(126000+100*depth+10*seed+method_index).normal(size=(64,16)))
                eval_axes=fit_axes(teacher,model,pilot,steps=200);evalq=DirectionalMixture(eval_axes,(32,128))
                evalrng=np.random.default_rng(124000+100*depth+seed)
                batches=[];inputs=[];allweights=[];values=[]
                for _ in range(6):
                    u,we=evalq.sample(8192,evalrng);xe=torch.from_numpy(4*u)
                    with torch.no_grad():
                        truth=teacher(xe);pred=model(xe)
                        value=np.column_stack([truth.square().mean(dim=1).numpy(),(pred-truth).square().mean(dim=1).numpy()])*we[:,None]
                    batches.append(value.mean(axis=0));inputs.append(xe.numpy());allweights.append(we);values.append(value)
                batches=np.array(batches);mean=batches.mean(axis=0);cv=batches.std(axis=0,ddof=1)/mean
                np.savez_compressed(out/f'{prefix}_evaluation.npz',inputs=np.concatenate(inputs),weights=np.concatenate(allweights),weighted_values=np.concatenate(values),axes=eval_axes)
                with torch.no_grad():train_loss=float(((model(x)-y).square().mean(dim=1)*weights).mean())
                row=dict(depth=depth,seed=seed,method=method,relative_mse=float(mean[1]/mean[0]),teacher_energy=float(mean[0]),error=float(mean[1]),batch_cv=cv.tolist(),
                         precision_gate=bool(np.all(cv<.1)),batches=batches.tolist(),training_loss=train_loss,clipped_steps=clipped,completed_steps=step+1,failure=failure,curve=curve,seconds=time.monotonic()-start)
                rows.append(row);(out/'results.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n')
                print(depth,seed,method,'relative',row['relative_mse'],'CV',row['batch_cv'],'gate',row['precision_gate'],flush=True)
    (out/'results.json').write_text(json.dumps(dict(status='complete',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
