"""CPU deep-polynomial smoke study; no population advantage claim."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask,independent_teacher
from homogeneous_evaluation import sphere_from_gaussian


def direct(model,x):
    outputs=[];i,j=np.triu_indices(model.h)
    for t in range(model.tasks):
        z=x
        for l,(a,b) in enumerate(zip(model.A,model.B)):
            a=a.detach().numpy();b=b.detach().numpy();s=getattr(model,'scale_'+str(l)).numpy()
            if not model.shared:a,b,s=a[t],b[t],s[t]
            z=(z@a.T)*(z@b.T)/s
        outputs.append((z[:,i]*z[:,j])@model.W[t].detach().numpy())
    return np.column_stack(outputs)


def main():
    torch.set_num_threads(1)
    root=Path(__file__).parent;out=root/'results/deep_smoke';out.mkdir(exist_ok=False)
    protocol=dict(d=16,h=8,tasks=6,depths=[2,3,4,5],degree='2**depth',steps=100,lr=.001,clip=10,
                  calibration=4096,training=256,evaluation=8192,distribution='sphere radius sqrt(d)',
                  seed=101000,teacher_seeds='102000+depth; independent task offsets100*(t+1)',student_seed=103000,
                  intent='correctness and numerical feasibility; one seed, short training, no transfer/advantage claim',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    rng=np.random.default_rng(101000)
    def sample(n):return torch.from_numpy(sphere_from_gaussian(rng.normal(size=(n,16))))
    cal,x,test=sample(4096),sample(256),sample(8192);rows=[];checks=[]
    for depth in [2,3,4,5]:
        shared=DeepMultiTask(16,8,6,depth,True,103000);separate=DeepMultiTask(16,8,6,depth,False,103000)
        shared.calibrate(cal);separate.calibrate(cal)
        torch.testing.assert_close(shared(x),separate(x),atol=1e-9,rtol=1e-9)
        for m in [shared,separate]:
            y=m(x[:17]);np.testing.assert_allclose(y.detach().numpy(),direct(m,x[:17].numpy()),atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(m(1.03*x[:17]),y*1.03**(2**depth),atol=1e-9,rtol=1e-9)
        g1=torch.autograd.grad(shared(x[:17]).sum(),tuple(shared.parameters()))
        g2=torch.autograd.grad(shared(x[:17],recompute=True).sum(),tuple(shared.parameters()))
        for a,b in zip(g1,g2):torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-9)
        checks.append(dict(depth=depth,degree=2**depth,matched_initial_functions=True,numpy_replay=True,homogeneity=True,recomputation_gradients=True))
        for kind in ['shared','independent']:
            teacher=(DeepMultiTask(16,8,6,depth,True,102000+depth) if kind=='shared' else independent_teacher(16,8,6,depth,102000+depth))
            teacher.calibrate(cal)
            with torch.no_grad():target=teacher(x);truth=teacher(test)
            for sharing in [True,False]:
                model=DeepMultiTask(16,8,6,depth,sharing,103000);model.calibrate(cal)
                opt=torch.optim.Adam(model.parameters(),lr=.001);start=time.monotonic();clipped=0;failure=None
                for step in range(100):
                    opt.zero_grad();loss=(model(x)-target).square().mean()
                    if not torch.isfinite(loss):failure='nonfinite loss';break
                    loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                    if not torch.isfinite(gn):failure='nonfinite gradient';break
                    clipped+=float(gn)>10;opt.step()
                with torch.no_grad():
                    err=(model(test)-truth).square().mean(dim=1).numpy();energy=truth.square().mean(dim=1).numpy()
                    train=float((model(x)-target).square().mean())
                row=dict(depth=depth,degree=2**depth,teacher=kind,shared_student=sharing,parameters=sum(p.numel() for p in model.parameters()),
                         train_mse=train,evaluation_mse=float(err.mean()),target_energy=float(energy.mean()),relative_mse=float(err.mean()/energy.mean()),
                         error_effective_samples=float(err.sum()**2/(err@err)),error_top_one_percent_fraction=float(np.sort(err)[-82:].sum()/err.sum()),
                         target_top_one_percent_fraction=float(np.sort(energy)[-82:].sum()/energy.sum()),
                         clipped_steps=clipped,completed_steps=step+1,failure=failure,seconds=time.monotonic()-start)
                rows.append(row);print(depth,kind,sharing,'relative MSE',row['relative_mse'],'error ESS',row['error_effective_samples'],flush=True)
                np.savez_compressed(out/f'd{depth}_{kind}_{sharing}.npz',squared_errors=err,target_energy=energy)
                (out/'results.json').write_text(json.dumps(dict(status='running',checks=checks,rows=rows),indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(dict(status='complete',checks=checks,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
