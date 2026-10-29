"""Full standard SVHN comparison; disjoint depth shards, recoverable epochs."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from vision_circuits import BilinearCircuit,nearest_tree_width
from nonuniform_tree import NonuniformTree,budget_widths

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--depth',type=int,choices=[2,3,4],required=True);args=parser.parse_args();torch.set_num_threads(1);root=Path(__file__).parent;out=root/f'results/svhn_nonuniform/depth{args.depth}';out.mkdir(parents=True,exist_ok=True);cache=Path('/home/loganriggs/.cache/shared_tensor_networks/svhn');manifest=json.loads((cache/'train_array_manifest.json').read_text())
    for key in ['images','labels']:assert digest(Path(manifest[key]['path']))==manifest[key]['sha256']
    images=np.load(manifest['images']['path'],mmap_mode='r');labels=np.load(manifest['labels']['path'],mmap_mode='r');assert images.shape==(73257,3,32,32) and labels.shape==(73257,)
    order=np.random.default_rng(200000).permutation(73257);train_ids=order[:68257];val_ids=order[68257:];total=np.zeros(3);squared=np.zeros(3);count=0
    for start in range(0,len(train_ids),512):
        a=images[train_ids[start:start+512]].astype('float64')/255;total+=a.sum(axis=(0,2,3));squared+=(a*a).sum(axis=(0,2,3));count+=len(a)*1024
    mean=total/count;std=np.sqrt(squared/count-mean*mean)
    def normalize(ids):
        a=images[ids].astype('float32');a/=255;a-=mean.astype('float32')[None,:,None,None];a/=std.astype('float32')[None,:,None,None];return torch.from_numpy(a.reshape(len(ids),3072))
    x=normalize(train_ids);vx=normalize(val_ids);y=torch.tensor(labels[train_ids]);vy=torch.tensor(labels[val_ids]);del images
    protocol=dict(dataset='SVHN standard train only',ntrain=68257,nvalidation=5000,split_seed=200000,model_seed=200100,depth=args.depth,width=32,level_widths=budget_widths(3072,32,args.depth),architecture='nonuniform_tree',epochs=30,batch_size=256,lr=.001,optimizer='Adam',clip=10,rmsnorm=True,mean=mean.tolist(),std=std.tolist(),preprocessing='RGB channel means/std from standard training only; flatten NCHW; no augmentation',official_test_used=False,extra_used=False,selection='fixed30epochs, full curves, validation for development only',source_sha256=digest(Path(__file__)),dependencies={n:digest(root/n) for n in ['vision_circuits.py','nonuniform_tree.py']},array_manifest_sha256=digest(cache/'train_array_manifest.json'))
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:write(out/'protocol.json',protocol);(out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'data_manifest.json').write_bytes((cache/'train_array_manifest.json').read_bytes());np.savez_compressed(out/'splits.npz',training=train_ids,validation=val_ids)
    def metrics(model,xx,yy):
        loss=0.;correct=0;pred=[]
        with torch.no_grad():
            for start in range(0,len(xx),512):
                z=model(xx[start:start+512]);assert torch.isfinite(z).all();loss+=float(nn.functional.cross_entropy(z,yy[start:start+512],reduction='sum'));correct+=int((z.argmax(1)==yy[start:start+512]).sum());pred.append(z)
        return dict(ce=loss/len(xx),accuracy=correct/len(xx)),torch.cat(pred)
    configs=['nonuniform_tree']
    for kind in configs:
        depth=0 if kind=='mlp' else args.depth;name=f'd{depth}_{kind}';cp=out/f'{name}_checkpoint.pt';start=time.monotonic()
        if kind=='mlp':
            torch.manual_seed(200100);model=nn.Sequential(nn.Linear(3072,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,10));width=64
        else:
            widths=budget_widths(3072,32,depth);width=32;model=NonuniformTree(3072,widths,10,True,200100)
        opt=torch.optim.Adam(model.parameters(),lr=.001);curves=[]
        if cp.exists():
            saved=torch.load(cp,weights_only=True);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);curves=saved['curves']
        with torch.no_grad():
            if not (out/f'{name}_initial.npy').exists():np.save(out/f'{name}_initial.npy',model(vx[:128]).numpy())
        write(out/'status.json',dict(status='running',pid=os.getpid(),configuration=name,completed_epochs=len(curves)))
        for epoch in range(len(curves),30):
            indices=torch.randperm(len(x),generator=torch.Generator().manual_seed(200200+epoch));clipped=0
            for batch in indices.split(256):
                opt.zero_grad();loss=nn.functional.cross_entropy(model(x[batch]),y[batch]);assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm);clipped+=int(norm>10);opt.step()
            train,_=metrics(model,x,y);validation,logits=metrics(model,vx,vy);curves.append(dict(epoch=epoch+1,train=train,validation=validation,clipped_steps=clipped,optimizer_updates=(epoch+1)*267,examples_seen=(epoch+1)*len(x)))
            tmp=cp.with_suffix('.tmp');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),curves=curves),tmp);tmp.replace(cp)
            row=dict(name=name,depth=depth,architecture=kind,rmsnorm=kind!='mlp',width=width,widths=widths,parameters=sum(p.numel() for p in model.parameters()),dense_multiply_accumulates_per_example=sum(p.numel() for n,p in model.named_parameters() if not n.endswith('bias')),curves=curves)
            write(out/f'{name}.json',row);write(out/'status.json',dict(status='running',pid=os.getpid(),configuration=name,completed_epochs=epoch+1));print(name,epoch+1,train,validation,flush=True)
        np.savez_compressed(out/f'{name}_logits.npz',validation=metrics(model,vx,vy)[1].numpy());write(out/f'{name}_runtime.json',dict(seconds_this_invocation=time.monotonic()-start))
    write(out/'status.json',dict(status='complete',configurations=len(configs)))
if __name__=='__main__':main()
