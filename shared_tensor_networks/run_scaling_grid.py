"""Representative scaling with strict source/heldout task separation."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask,independent_teacher
from directional_sampling import DirectionalMixture
from homogeneous_evaluation import sphere_from_gaussian


def count(d,h,depth,shared,tasks=12):
    trunk=2*d*h+2*(depth-2)*h*h
    return trunk*(1 if shared else tasks)+tasks*h*(h+1)//2


def budget_width(d,h,depth):
    target=count(d,h,depth,False)
    return min(range(1,4*h+1),key=lambda width:abs(count(d,width,depth,True)-target))


def proposal(target,predict,d,seed,steps=100):
    """Pilot axes alternate target-energy and residual objectives.

    Target/predict must expose source outputs only until final evaluation.
    Separate logs prevent target energy from hiding small residuals.
    """
    x=torch.from_numpy(np.random.default_rng(seed).normal(size=(32,d))).requires_grad_(True)
    for _ in range(steps):
        u=x/torch.linalg.vector_norm(x,dim=1,keepdim=True)*math.sqrt(d)
        y=target(u);e=predict(u)-y
        value=torch.cat([y[:16].square().mean(dim=1),e[16:].square().mean(dim=1)])
        grad,=torch.autograd.grad(torch.log(value+1e-30).sum(),x)
        with torch.no_grad():
            x+=.15*grad/torch.linalg.vector_norm(grad,dim=1,keepdim=True).clamp_min(1e-20)
            x/=torch.linalg.vector_norm(x,dim=1,keepdim=True)
    axes=x.detach().numpy()
    return DirectionalMixture(axes,(16,64)),axes


def draws(q,d,n,seed):
    u,w=q.sample(n,np.random.default_rng(seed))
    return torch.from_numpy(u*math.sqrt(d)),torch.from_numpy(w)


def rms(model,q,d,seed):
    sums=[]
    for j in range(4):
        x,w=draws(q,d,4096,seed+j)
        with torch.no_grad():sums.append((model(x).square()*w[:,None]).mean(dim=0))
    return torch.stack(sums).mean(dim=0).sqrt().clamp_min(1e-12)


def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/scaling_grid';out.mkdir(exist_ok=False)
    grid=[(16,8,2),(16,8,3),(32,16,2),(32,16,3)]
    protocol=dict(grid=grid,teachers=['shared','independent'],replicates=3,models=['shared','separate','shared_budget'],expected_fits=72,
                  source_tasks=list(range(12)),heldout_tasks=list(range(12,16)),steps=600,ntrain=1024,optimizer='Adam',lr=.001,clip=10,
                  pilot_axes=32,pilot_steps=100,stretches=[16,64],calibration_samples=16384,evaluation_batches=6,evaluation_batch_size=8192,
                  adaptation_labels=[32,128],ridge=.001,transfer_features='12 predicted source outputs normalized on independent source-only calibration',
                  baseline='true source outputs with same adaptation data and ridge',
                  rng='base=2000000+10000000*grid_index+1000000*regime_index+10000*replicate; offsets specified in frozen source',
                  heldout_policy='heldout outputs never influence source proposal, student initialization, source loss, optimizer or checkpoint selection; only adaptation fitting and final evaluation',
                  teacher_calibration='all fixed teacher heads normalized using source-only proposal; head scaling is separate, no teacher training',
                  limitations='privileged source-teacher gradient proposal and calibration; not equal-total-supervision sample efficiency; three teacher replicates; different classes/metrics',
                  budgets=[dict(d=d,h=h,depth=l,wide=budget_width(d,h,l),shared=count(d,h,l,True),separate=count(d,h,l,False),shared_budget=count(d,budget_width(d,h,l),l,True)) for d,h,l in grid],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dependencies={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['deep_multitask.py','directional_sampling.py','homogeneous_evaluation.py']})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());rows=[]
    for gi,(d,h,depth) in enumerate(grid):
        for ki,kind in enumerate(['shared','independent']):
            for rep in range(3):
                base=2000000+10000000*gi+1000000*ki+10000*rep;key=f'g{gi}_{kind}_r{rep}';folder=out/key;folder.mkdir()
                teacher=(DeepMultiTask(d,h,16,depth,True,base) if kind=='shared' else independent_teacher(d,h,16,depth,base))
                cal=torch.from_numpy(sphere_from_gaussian(np.random.default_rng(base+1).normal(size=(4096,d))));teacher.calibrate(cal)
                source=lambda x:teacher(x)[:,:12]
                tq,taxes=proposal(source,lambda x:source(x)*0,d,base+2)
                trms=rms(teacher,tq,d,base+3)
                with torch.no_grad():teacher.W.div_(trms[:,None])
                torch.save(teacher.state_dict(),folder/'teacher.pt');np.savez_compressed(folder/'teacher_calibration.npz',axes=taxes,rms=trms.numpy())
                train,weights=draws(tq,d,1024,base+7)
                with torch.no_grad():labels=source(train)
                adapt=torch.from_numpy(sphere_from_gaussian(np.random.default_rng(base+8).normal(size=(128,d))))
                with torch.no_grad():ay=teacher(adapt)[:,12:].numpy()
                np.savez_compressed(folder/'training.npz',inputs=train.numpy(),weights=weights.numpy(),targets=labels.numpy(),adaptation_inputs=adapt.numpy(),adaptation_targets=ay)
                oracle_norm=rms(source,tq,d,base+9).numpy()
                with torch.no_grad():oracle_features=source(adapt).numpy()/oracle_norm
                def ridge(Z,Y):return np.linalg.solve(Z.T@Z/len(Z)+.001*np.eye(12),Z.T@Y/len(Z))
                oracle_B=[ridge(oracle_features[:k],ay[:k]) for k in [32,128]]
                reference=None
                for mi,(name,width,shared) in enumerate([('shared',h,True),('separate',h,False),('shared_budget',budget_width(d,h,depth),True)]):
                    model=DeepMultiTask(d,width,12,depth,shared,base+13);model.calibrate(cal)
                    if name=='separate':
                        maxes=reference_axes.copy();mq=DirectionalMixture(maxes,(16,64));mrms=reference_rms.clone()
                    else:
                        mq,maxes=proposal(model,lambda x:model(x)*0,d,base+14)
                        mrms=rms(model,mq,d,base+15)
                        if name=='shared':reference_axes=maxes.copy();reference_rms=mrms.clone()
                    with torch.no_grad():model.W.div_(mrms[:,None])
                    if name=='shared':reference=model(train).detach()
                    if name=='separate':torch.testing.assert_close(model(train),reference,rtol=1e-9,atol=1e-9)
                    assert sum(p.numel() for p in model.parameters())==count(d,width,depth,shared)
                    torch.save(model.state_dict(),folder/f'{name}_initial.pt')
                    np.savez_compressed(folder/f'{name}_calibration.npz',axes=maxes,rms=mrms.numpy())
                    opt=torch.optim.Adam(model.parameters(),lr=.001);start=time.monotonic();clipped=0;failure=None;curve=[]
                    for step in range(600):
                        opt.zero_grad();loss=((model(train)-labels).square().mean(dim=1)*weights).mean()
                        if not torch.isfinite(loss):failure='nonfinite loss';break
                        loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                        if not torch.isfinite(gn):failure='nonfinite gradient';break
                        clipped+=float(gn)>10;opt.step()
                        if (step+1)%100==0:curve.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(gn)))
                    torch.save(model.state_dict(),folder/f'{name}.pt')
                    if failure:
                        rows.append(dict(key=key,model=name,failure=failure));(out/'results.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n');continue
                    # Source-only normalizer for transfer, independent from adaptation/evaluation.
                    fq,faxes=proposal(source,model,d,base+20+mi)
                    fnorm=rms(model,fq,d,base+23+4*mi).numpy()
                    with torch.no_grad():features=model(adapt).numpy()/fnorm
                    Bs=[ridge(features[:k],ay[:k]) for k in [32,128]]
                    # Heldout target access begins only in this evaluation stage.
                    def target(x):
                        y=teacher(x);return torch.cat([y[:,:12],y[:,12:],y[:,12:],y[:,12:],y[:,12:]],dim=1)
                    def predict(x):
                        src=model(x);f=src/torch.from_numpy(fnorm);oracle=source(x)/torch.from_numpy(oracle_norm)
                        return torch.cat([src]+[f@torch.from_numpy(b) for b in Bs]+[oracle@torch.from_numpy(b) for b in oracle_B],dim=1)
                    eq,eaxes=proposal(target,predict,d,base+40+mi)
                    batches=[];metrics=[]
                    for batch in range(6):
                        xx,ww=draws(eq,d,8192,base+50+10*mi+batch)
                        with torch.no_grad():yy=target(xx).numpy();pp=predict(xx).numpy()
                        # Preserve each heldout task separately:12 source aggregate +4x4 transfer records.
                        delta=(pp-yy)**2;energy=yy**2
                        values=np.column_stack([delta[:,:12].mean(axis=1),energy[:,:12].mean(axis=1),delta[:,12:],energy[:,12:]])*ww.numpy()[:,None]
                        batches.append(values.mean(axis=0));metrics.append(values)
                    batches=np.array(batches);mean=batches.mean(axis=0);cv=batches.std(axis=0,ddof=1)/np.maximum(np.abs(mean),1e-300)
                    np.savez_compressed(folder/f'{name}_evaluation.npz',axes=eaxes,feature_axes=faxes,feature_norm=fnorm,oracle_norm=oracle_norm,B=np.array(Bs),oracle_B=np.array(oracle_B),weighted_values=np.concatenate(metrics))
                    with torch.no_grad():trainloss=float(((model(train)-labels).square().mean(dim=1)*weights).mean())
                    row=dict(key=key,grid_index=gi,d=d,h=h,depth=depth,teacher=kind,replicate=rep,model=name,width=width,parameters=count(d,width,depth,shared),failure=None,completed_steps=step+1,train_mse=trainloss,source_relative_mse=float(mean[0]/mean[1]),source_precision=bool(np.all(cv[:2]<.1)),transfer_relative_mse=(mean[2:18]/mean[18:34]).reshape(4,4).tolist(),metric_batch_cv=cv.tolist(),batches=batches.tolist(),curve=curve,clipped_steps=clipped,seconds=time.monotonic()-start)
                    rows.append(row);(out/'results.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n');print(key,name,'fits',len(rows),'source',row['source_relative_mse'],flush=True)
    (out/'results.json').write_text(json.dumps(dict(status='complete',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
