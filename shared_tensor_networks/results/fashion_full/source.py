"""Full-pool, longer-training development experiment with per-epoch recovery."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fashion_data import load
from vision_circuits import BilinearCircuit,nearest_tree_width


def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/fashion_full';out.mkdir(exist_ok=True)
    images,labels,_,_=load('/home/loganriggs/.cache/shared_tensor_networks/fashion-mnist')
    order=np.random.default_rng(150000).permutation(60000);train_ids=order[:55000];val_ids=order[55000:]
    raw=torch.from_numpy(images[train_ids].astype('float32')/255);mean=raw.mean();std=raw.std(unbiased=False);x=(raw-mean)/std;y=torch.from_numpy(labels[train_ids].astype('int64'))
    vx=(torch.from_numpy(images[val_ids].astype('float32')/255)-mean)/std;vy=torch.from_numpy(labels[val_ids].astype('int64'))
    protocol=dict(ntrain=55000,nvalidation=5000,split_seed=150000,model_seed=150100,epochs=30,batch_size=256,lr=.001,optimizer='Adam',clip=10,depths=[2,3,4],width=32,rmsnorm=True,mean=float(mean),std=float(std),official_test_used=False,selection='fixed 30 epochs; validation development only; convergence is measured, not assumed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dependencies={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['vision_circuits.py','fashion_data.py']})
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:
        write_json(out/'protocol.json',protocol);(out/'source.py').write_bytes(Path(__file__).read_bytes());np.savez_compressed(out/'splits.npz',training=train_ids,validation=val_ids)
    def metrics(model,xx,yy):
        total=0.;correct=0;logits=[]
        with torch.no_grad():
            for a in range(0,len(xx),512):
                z=model(xx[a:a+512]);assert torch.isfinite(z).all();total+=float(nn.functional.cross_entropy(z,yy[a:a+512],reduction='sum'));correct+=int((z.argmax(1)==yy[a:a+512]).sum());logits.append(z)
        return dict(ce=total/len(xx),accuracy=correct/len(xx)),torch.cat(logits)
    configs=[(d,k) for d in [2,3,4] for k in ['stack','tree','tree_budget']]+[(0,'mlp')]
    for depth,kind in configs:
        name=f'd{depth}_{kind}';cp=out/f'{name}_checkpoint.pt';start=time.monotonic()
        if kind=='mlp':
            torch.manual_seed(150100);model=nn.Sequential(nn.Linear(784,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,10));width=64
        else:
            width=nearest_tree_width(784,32,depth) if kind=='tree_budget' else 32
            model=BilinearCircuit(784,width,depth,10,kind!='stack',True,150100)
        opt=torch.optim.Adam(model.parameters(),lr=.001);curves=[]
        if cp.exists():
            saved=torch.load(cp,weights_only=True);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);curves=saved['curves']
        write_json(out/'status.json',dict(status='running',pid=os.getpid(),configuration=name,completed_epochs=len(curves)))
        for epoch in range(len(curves),30):
            ids=torch.randperm(len(x),generator=torch.Generator().manual_seed(150200+epoch));clipped=0
            for batch in ids.split(256):
                opt.zero_grad();loss=nn.functional.cross_entropy(model(x[batch]),y[batch]);assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(gn);clipped+=int(gn>10);opt.step()
            train,_=metrics(model,x,y);val,logits=metrics(model,vx,vy)
            curves.append(dict(epoch=epoch+1,train=train,validation=val,clipped_steps=clipped))
            tmp=cp.with_suffix('.tmp');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),curves=curves),tmp);tmp.replace(cp)
            row=dict(name=name,depth=depth,architecture=kind,rmsnorm=kind!='mlp',width=width,parameters=sum(p.numel() for p in model.parameters()),curves=curves)
            write_json(out/f'{name}.json',row);print(name,epoch+1,train,val,flush=True)
        np.savez_compressed(out/f'{name}_logits.npz',validation=metrics(model,vx,vy)[1].numpy())
        write_json(out/f'{name}_runtime.json',dict(seconds_this_invocation=time.monotonic()-start))
    write_json(out/'status.json',dict(status='complete',configurations=len(configs)))
if __name__=='__main__':main()
