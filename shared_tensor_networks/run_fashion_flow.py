"""Shared versus projected/free SGD: full-pool one-epoch dynamics control."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fashion_data import load
from vision_circuits import BilinearCircuit

def groups(model):
    yield 'W',[model.W]
    for prefix in ['A','B']:
        for level in range(model.depth):
            yield f'{prefix}.{level}',[getattr(model,prefix)[i] for i,l in enumerate(model.levels) if l==level]

def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/fashion_flow';out.mkdir(exist_ok=True)
    protocol=dict(depths=[2,3,4],seeds=[160001,160002,160003],width=16,ntrain=55000,nvalidation=5000,epochs=1,batch=256,lr=.01,dtype='float64',rmsnorm=True,optimizer='plain SGD, no momentum, decay or clipping',arms=['ordinary','corrected','projected','free'],purpose='dynamics control, not converged accuracy comparison',official_test_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    images,labels,_,_=load('/home/loganriggs/.cache/shared_tensor_networks/fashion-mnist');ids=np.random.default_rng(150000).permutation(60000)
    raw=torch.from_numpy(images[ids[:55000]].astype('float64')/255);mean=raw.mean();std=raw.std(unbiased=False);x=(raw-mean)/std;del raw
    y=torch.from_numpy(labels[ids[:55000]].astype('int64'));vx=(torch.from_numpy(images[ids[55000:]].astype('float64')/255)-mean)/std;vy=torch.from_numpy(labels[ids[55000:]].astype('int64'))
    def loss(m,a,b):return nn.functional.cross_entropy(m(a),b)
    def metrics(m):
        with torch.no_grad():z=torch.cat([m(a) for a in vx.split(256)]);return dict(ce=float(nn.functional.cross_entropy(z,vy)),accuracy=float((z.argmax(1)==vy).double().mean()))
    for depth in [2,3,4]:
        for seed in protocol['seeds']:
            name=f'd{depth}_s{seed}';dest=out/f'{name}.json'
            if dest.exists():continue
            start=time.monotonic();models={k:BilinearCircuit(784,16,depth,10,k in ['projected','free'],True,seed,torch.float64) for k in protocol['arms']}
            shared=models['ordinary'];tree=models['free'];px=x[:256];py=y[:256]
            torch.testing.assert_close(shared(px),tree(px),rtol=1e-12,atol=1e-12)
            loss(shared,px,py).backward();loss(tree,px,py).backward()
            sg=dict(groups(shared));projected_energy=0.;free_energy=0.;disagreement=0.;layer=[]
            for key,ps in groups(tree):
                gradients=torch.stack([p.grad for p in ps]);avg=gradients.mean(0)
                torch.testing.assert_close(sg[key][0].grad,gradients.sum(0),rtol=1e-10,atol=1e-10)
                pe=len(ps)*float(avg.square().sum());fe=float(gradients.square().sum());de=float((gradients-avg).square().sum());projected_energy+=pe;free_energy+=fe;disagreement+=de
                layer.append(dict(parameter=key,multiplicity=len(ps),free=fe,projected=pe,disagreement=de))
            assert abs(free_energy-projected_energy-disagreement)<1e-9*max(1,free_energy)
            ordinary_energy=sum(float(p.grad.square().sum()) for p in shared.parameters())
            # Independently verify all three predicted directional derivatives by central differences.
            directional=[]
            for arm in ['ordinary','corrected','free']:
                m=models['free'] if arm=='free' else shared;params=list(m.parameters());original=[p.detach().clone() for p in params];direction=[]
                for n,p in m.named_parameters():
                    multiplicity=1 if n=='W' or arm!='corrected' else 2**(depth-1-int(n.split('.')[1]))
                    direction.append(-p.grad/multiplicity)
                values=[];eps=1e-6
                with torch.no_grad():
                    for sign in [1,-1]:
                        for p,o,v in zip(params,original,direction):p.copy_(o+sign*eps*v)
                        values.append(float(loss(m,px,py)))
                    for p,o in zip(params,original):p.copy_(o)
                actual=(values[0]-values[1])/(2*eps);expected=-dict(ordinary=ordinary_energy,corrected=projected_energy,free=free_energy)[arm]
                assert abs(actual-expected)<2e-6*max(1,abs(expected)),(arm,actual,expected)
                directional.append(dict(arm=arm,finite_difference=actual,predicted=expected))
            for m in models.values():m.zero_grad(set_to_none=True)
            initial={k:metrics(m) for k,m in models.items()};trace=[];max_equivalence=0.
            order=torch.randperm(len(x),generator=torch.Generator().manual_seed(160100))
            for step,batch in enumerate(order.split(256),1):
                for arm,m in models.items():
                    m.zero_grad(set_to_none=True);ell=loss(m,x[batch],y[batch]);assert torch.isfinite(ell);ell.backward()
                    with torch.no_grad():
                        for n,p in m.named_parameters():
                            mult=2**(depth-1-int(n.split('.')[1])) if arm=='corrected' and n!='W' else 1
                            assert torch.isfinite(p.grad).all();p.add_(p.grad,alpha=-.01/mult)
                        if arm=='projected':
                            for _,ps in groups(m):
                                avg=torch.stack(ps).mean(0)
                                for p in ps:p.copy_(avg)
                if step%25==0 or step==215:
                    cp=dict(groups(models['corrected']));err=max(float((p-cp[k][0]).abs().max()) for k,ps in groups(models['projected']) for p in ps);max_equivalence=max(max_equivalence,err);assert err<1e-8
                    trace.append(dict(step=step,validation={k:metrics(m) for k,m in models.items()},corrected_projected_max_parameter_error=err))
            torch.save({k:m.state_dict() for k,m in models.items()},out/f'{name}.pt')
            result=dict(depth=depth,seed=seed,initial=initial,probe=dict(free_energy=free_energy,projected_energy=projected_energy,disagreement_energy=disagreement,disagreement_fraction=disagreement/free_energy,ordinary_energy=ordinary_energy,layers=layer,directional_checks=directional),trace=trace,max_equivalence_error=max_equivalence,seconds=time.monotonic()-start)
            dest.write_text(json.dumps(result,indent=2)+'\n');print(name,'disagreement_fraction',disagreement/free_energy,'equivalence_error',max_equivalence,'final',trace[-1]['validation'],flush=True)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',cases=9))+'\n')
if __name__=='__main__':main()
