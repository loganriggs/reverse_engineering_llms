"""CPU development pilot; official test set is not evaluated."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fashion_data import load
from vision_circuits import BilinearCircuit,nearest_tree_width,parameter_count


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/fashion_pilot'
    if not args.resume:out.mkdir(exist_ok=False)
    else:assert (out/'protocol.json').exists()
    cache=Path('/home/loganriggs/.cache/shared_tensor_networks/fashion-mnist')
    images,labels,_,_=load(cache)
    order=np.random.default_rng(150000).permutation(60000);train_ids=order[:2048];val_ids=order[55000:]
    assert not set(train_ids).intersection(val_ids)
    raw=torch.from_numpy(images[train_ids].astype('float32')/255);mean=raw.mean();std=raw.std(unbiased=False)
    x=(raw-mean)/std;y=torch.from_numpy(labels[train_ids].astype('int64'))
    vx=(torch.from_numpy(images[val_ids].astype('float32')/255)-mean)/std;vy=torch.from_numpy(labels[val_ids].astype('int64'))
    protocol=dict(dataset='Fashion-MNIST official train split only; official test untouched',ntrain=2048,nvalidation=5000,split_seed=150000,model_seed=150100,epochs=5,batch_size=128,lr=.001,optimizer='Adam',clip=10,
                  depths=[2,3,4],width=32,rmsnorm=[False,True],norm='per-node mean-square, eps1e-6, no learned gain',
                  architectures=['shared stack','fully untied matched-width binary tree','near-budget narrower tree','one two-hidden-layer64-ReLU MLP baseline'],
                  preprocessing='pixels/255; scalar mean/std fitted on training subset only; no augmentation',mean=float(mean),std=float(std),
                  selection='fixed5epochs; no early stopping/best checkpoint; validation is development evidence, not test performance',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependencies={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['vision_circuits.py','fashion_data.py']})
    if args.resume:
        frozen=json.loads((out/'protocol.json').read_text())
        for key,value in protocol.items():
            if key!='source_sha256':assert frozen[key]==value,(key,frozen[key],value)
        previous=json.loads((out/'results.json').read_text());rows=previous['rows']
        assert len({r['name'] for r in rows})==len(rows)
        recovery=dict(previous_status=previous['status'],previous_results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),resume_source_sha256=protocol['source_sha256'],completed_names=[r['name'] for r in rows])
        (out/'interrupted_results.json').write_bytes((out/'results.json').read_bytes())
        (out/'resume_source.py').write_bytes(Path(__file__).read_bytes())
        (out/'recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
    else:
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'dataset_manifest.json').write_bytes((cache/'manifest.json').read_bytes());np.savez_compressed(out/'splits.npz',training=train_ids,validation=val_ids)
        rows=[]
    configs=[(depth,norm,kind) for depth in [2,3,4] for norm in [False,True] for kind in ['stack','tree','tree_budget']]+[(0,False,'mlp')]
    for depth,norm,kind in configs:
        name=f'd{depth}_norm{int(norm)}_{kind}'
        if name in {r['name'] for r in rows}:continue
        if kind=='mlp':
            torch.manual_seed(150100);model=nn.Sequential(nn.Linear(784,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,10));width=64
        else:
            width=nearest_tree_width(784,32,depth) if kind=='tree_budget' else 32
            model=BilinearCircuit(784,width,depth,10,kind!='stack',norm,150100)
            assert sum(p.numel() for p in model.parameters())==parameter_count(784,width,depth,10,kind!='stack')
        opt=torch.optim.Adam(model.parameters(),lr=.001);start=time.monotonic();curves=[];failure=None;clipped=0
        with torch.no_grad():initial_logits=model(vx[:128]).numpy()
        for epoch in range(5):
            ids=torch.randperm(len(x),generator=torch.Generator().manual_seed(150200+epoch))
            for batch in ids.split(128):
                opt.zero_grad();pred=model(x[batch]);loss=nn.functional.cross_entropy(pred,y[batch])
                if not torch.isfinite(loss):failure='nonfinite training loss';break
                loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(gn):failure='nonfinite gradient';break
                clipped+=float(gn)>10;opt.step()
            if failure:break
            with torch.no_grad():
                logits=torch.cat([model(z) for z in vx.split(256)]);train_logits=model(x)
                if not torch.isfinite(logits).all():failure='nonfinite validation logits';break
                curves.append(dict(epoch=epoch+1,train_ce=float(nn.functional.cross_entropy(train_logits,y)),train_accuracy=float((train_logits.argmax(1)==y).float().mean()),validation_ce=float(nn.functional.cross_entropy(logits,vy)),validation_accuracy=float((logits.argmax(1)==vy).float().mean())))
        torch.save(model.state_dict(),out/f'{name}.pt')
        if failure is None:np.savez_compressed(out/f'{name}_logits.npz',initial=initial_logits,validation=logits.numpy())
        row=dict(name=name,depth=depth,rmsnorm=norm,architecture=kind,width=width,parameters=sum(p.numel() for p in model.parameters()),failure=failure,curves=curves,clipped_steps=clipped,seconds=time.monotonic()-start)
        rows.append(row);(out/'results.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2)+'\n');print(name,'params',row['parameters'],'failure',failure,'final',curves[-1] if curves else None,flush=True)
    (out/'results.json').write_text(json.dumps(dict(status='complete',rows=rows,max_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')


if __name__=='__main__':main()
