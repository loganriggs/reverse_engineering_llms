"""Match initial loss descent (not output speed) using training-only calibration."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fashion_data import load
from vision_circuits import BilinearCircuit

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def energies(shared,tree,x,y):
    for m in [shared,tree]:m.zero_grad(set_to_none=True);nn.functional.cross_entropy(m(x),y).backward()
    ordinary=sum(float(p.grad.square().sum()) for p in shared.parameters());corrected=0
    for n,p in shared.named_parameters():corrected+=float(p.grad.square().sum())/(1 if n=='W' else 2**(shared.depth-1-int(n.split('.')[1])))
    free=sum(float(p.grad.square().sum()) for p in tree.parameters())
    return dict(ordinary=ordinary,corrected=corrected,free=free)

def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/fashion_flow_rates';out.mkdir(exist_ok=True);baseline=root/'results/fashion_flow'
    protocol=dict(depths=[2,3,4],seeds=[160001,160002,160003],width=16,ntrain=55000,nvalidation=5000,epochs=1,batch=256,dtype='float64',rmsnorm=True,base_lr=.01,calibration='first256training; lr_arm=.01*energy_ordinary/energy_arm; fixed thereafter',probe_only='next3disjoint256training batches, not used to select rates',optimizer='SGD no momentum/clipping/decay',source_sha256=digest(Path(__file__)),dependencies={n:digest(root/n) for n in ['vision_circuits.py','fashion_data.py']},baseline_protocol_sha256=digest(baseline/'protocol.json'),official_test_used=False)
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    images,labels,_,_=load('/home/loganriggs/.cache/shared_tensor_networks/fashion-mnist');ids=np.random.default_rng(150000).permutation(60000);raw=torch.from_numpy(images[ids[:55000]].astype('float64')/255);mean=raw.mean();std=raw.std(unbiased=False);x=(raw-mean)/std;del raw
    y=torch.from_numpy(labels[ids[:55000]].astype('int64'));vx=(torch.from_numpy(images[ids[55000:]].astype('float64')/255)-mean)/std;vy=torch.from_numpy(labels[ids[55000:]].astype('int64'))
    def metrics(m):
        with torch.no_grad():z=torch.cat([m(a) for a in vx.split(256)]);return dict(ce=float(nn.functional.cross_entropy(z,vy)),accuracy=float((z.argmax(1)==vy).double().mean()))
    for depth in protocol['depths']:
        for seed in protocol['seeds']:
            name=f'd{depth}_s{seed}';dest=out/f'{name}.json'
            if dest.exists():continue
            start=time.monotonic();models={a:BilinearCircuit(784,16,depth,10,a=='free',True,seed,torch.float64) for a in ['corrected','free']}
            torch.testing.assert_close(models['corrected'](x[:256]),models['free'](x[:256]),rtol=1e-12,atol=1e-12)
            probes=[energies(models['corrected'],models['free'],x[i:i+256],y[i:i+256]) for i in range(0,1024,256)]
            ref=json.loads((baseline/f'{name}.json').read_text());e=probes[0]
            for a,key in [('ordinary','ordinary_energy'),('corrected','projected_energy'),('free','free_energy')]:assert abs(e[a]-ref['probe'][key])<1e-9*max(1,e[a])
            rates={a:.01*e['ordinary']/e[a] for a in models};checks=[{a:rates[a]*q[a]/(.01*q['ordinary']) for a in models} for q in probes]
            traces=[];failures={};order=torch.randperm(len(x),generator=torch.Generator().manual_seed(160100))
            for step,batch in enumerate(order.split(256),1):
                for arm,m in models.items():
                    if arm in failures:continue
                    m.zero_grad(set_to_none=True);ell=nn.functional.cross_entropy(m(x[batch]),y[batch])
                    if not torch.isfinite(ell):failures[arm]=dict(step=step,reason='nonfinite loss');continue
                    ell.backward()
                    if not all(torch.isfinite(p.grad).all() for p in m.parameters()):failures[arm]=dict(step=step,reason='nonfinite gradient');continue
                    with torch.no_grad():
                        for n,p in m.named_parameters():
                            mult=2**(depth-1-int(n.split('.')[1])) if arm=='corrected' and n!='W' else 1
                            p.add_(p.grad,alpha=-rates[arm]/mult)
                if step%25==0 or step==215:traces.append(dict(step=step,validation={a:metrics(m) for a,m in models.items() if a not in failures}))
            torch.save({a:m.state_dict() for a,m in models.items()},out/f'{name}.pt')
            record=dict(depth=depth,seed=seed,rates=rates,probe_energies=probes,relative_initial_descent=checks,trace=traces,failures=failures,baseline_result_sha256=digest(baseline/f'{name}.json'),seconds=time.monotonic()-start)
            dest.write_text(json.dumps(record,indent=2)+'\n');print(name,'rates',rates,'final',traces[-1],'failures',failures,flush=True)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',cases=9))+'\n')
if __name__=='__main__':main()
