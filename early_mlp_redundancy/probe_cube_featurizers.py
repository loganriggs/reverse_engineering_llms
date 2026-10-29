"""M109 fixed MLP1 featurizers on held-out repetition-factor response geometry."""
import json
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    pp=V/'cube_featurizer_protocol.json';p=json.loads(pp.read_text());assert p['producer_sha256']==sha(__file__)
    for name,h in p['bindings'].items():assert sha(V/name)==h
    nr=json.loads((V/'repetition_cube_native.json').read_text());assert sha(nr['states_path'])==nr['states_sha256']
    old=torch.load(nr['states_path'],weights_only=True)
    rows=[r for r in json.loads((V/'repetition_cube_protocol.json').read_text())['rows'] if r['cohort']!='historical_replay']
    models=[]
    for e in p['models']:
        assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True)
        f=Featurizer('block',e['seed']).cuda();f.load_state_dict(s['state']);D=torch.nn.functional.normalize(f.dictionary,dim=0)
        models.append((e,f,s['mean'].cuda(),s['scale'].cuda(),D,torch.linalg.pinv(D.double())))
    path=A/'emlp_m109_cube_featurizers.pt';dest=V/'cube_featurizer_native.json';assert not path.exists() and not dest.exists()
    model,mr=load_model();assert mr.weights_sha256==nr['checkpoint_sha256']
    results={};forwards=0
    for r in rows:
        ids=torch.tensor([r['token_ids']],device='cuda');base,cap=trace(model,ids);forwards+=1
        assert torch.equal(base[0,-1].cpu(),old[r['id']]['logits'])
        w=old[r['id']]['layers'][1]['write'][0,-1].cuda()
        for e,f,mu,scale,D,pinv in models:
          x=(w-mu)/scale
          sparse,q,code=f(x)
          dense=(D.double()@(pinv@(x.double()-f.bias.double()))+f.bias.double()).float()
          for mode,rec in [('sparse',sparse),('dense',dense)]:
            reconstructed=rec*scale+mu
            def replace(i,z,y):
                if i!=1:return y
                assert torch.equal(z,cap[1]['normalized_input'])
                out=y.clone();out[0,-1]=reconstructed+model.transformer.h[1].mlp.Down_bias
                assert torch.equal(out[:,:-1],y[:,:-1]);return out
            edited,cc=trace(model,ids,replace=replace);forwards+=1
            key=f"{r['id']}|{e['seed']}|{e['arm']}|{mode}"
            results[key]=dict(logits=edited[0,-1].cpu(),write=reconstructed.cpu(),effective_write=cc[1]['effective_write'][0,-1].cpu())
        if forwards%72==0:print('FORWARDS',forwards,flush=True)
    torch.save(results,path)
    out=dict(status='completed_pending_independent_audit',protocol_sha256=sha(pp),source_sha256=sha(__file__),path=str(path),sha256=sha(path),forward_count=forwards,baseline_replays=len(rows),checkpoint_sha256=mr.weights_sha256)
    dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
