"""M109 frozen native cubic-component removal and signed-permutation controls."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace
from repetition_cube_algebra import H


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.set_grad_enabled(False)
    pp=V/'cube_causal_protocol.json';p=json.loads(pp.read_text())
    assert p['producer_sha256']==sha(__file__)
    for name,digest in p['bindings'].items():assert sha(V/name)==digest,name
    receipt=json.loads((V/'repetition_cube_native.json').read_text())
    assert sha(receipt['states_path'])==receipt['states_sha256']
    old=torch.load(receipt['states_path'],weights_only=True)
    rows=json.loads((V/'repetition_cube_protocol.json').read_text())['rows']
    rows=[r for r in rows if r['cohort']!='historical_replay']
    outpath=A/'emlp_m109_cube_causal.pt';dest=V/'cube_causal_native.json'
    assert not outpath.exists() and not dest.exists()
    model,mr=load_model();assert mr.weights_sha256==receipt['checkpoint_sha256']
    results={};directions={};forwards=0;replays=0;maximum=0.
    for case in range(4):
      for fmt in ['semicolon','comma']:
        items=sorted([r for r in rows if r['case']==case and r['format']==fmt],key=lambda r:r['vertex'])
        assert len(items)==8
        for layer in p['layers']:
          writes=torch.stack([old[r['id']]['layers'][layer]['write'][0,-1].double() for r in items])
          c=torch.tensor(H[:,7],dtype=torch.float64)@writes/8
          ds={'remove_cubic':c}
          for seed in p['scramble_seeds']:
            gen=np.random.default_rng(seed+layer*100)
            perm=gen.permutation(c.numel());sign=gen.choice([-1,1],c.numel())
            ds['scramble_'+str(seed)]=c[torch.tensor(perm)]*torch.tensor(sign)
          directions[f'{case}|{fmt}|{layer}']=ds
          for d in ds.values():assert torch.allclose(d.norm(),c.norm(),rtol=1e-12,atol=1e-12)
        for r in items:
          ids=torch.tensor([r['token_ids']],device='cuda')
          baseline,cap=trace(model,ids);forwards+=1
          assert torch.equal(baseline[0,-1].cpu(),old[r['id']]['logits']);replays+=1
          for layer in p['layers']:
            assert torch.equal(cap[layer]['normalized_input'].cpu(),old[r['id']]['layers'][layer]['input'])
            for arm,d in directions[f'{case}|{fmt}|{layer}'].items():
              delta=(-int(H[r['vertex'],7])*d).float().cuda()
              def replace(i,z,y):
                if i!=layer:return y
                assert torch.equal(z,cap[layer]['normalized_input'])
                result=y.clone();result[0,-1]+=delta
                assert torch.equal(result[:,:-1],y[:,:-1])
                return result
              changed,cc=trace(model,ids,replace=replace);forwards+=1
              effective=(cc[layer]['effective_write']-cc[layer]['native_write'])[0,-1]
              err=float((effective-delta).norm()/delta.norm().clamp_min(1e-12));maximum=max(maximum,err)
              assert err<1e-4
              results[f"{r['id']}|{layer}|{arm}"]=dict(logits=changed[0,-1].cpu(),effective_delta=effective.cpu())
        print('CUBE',case,fmt,'FORWARDS',forwards,flush=True)
    torch.save(dict(results=results,directions=directions),outpath)
    result=dict(status='completed_pending_independent_audit',protocol_sha256=sha(pp),producer_sha256=sha(__file__),checkpoint_sha256=mr.weights_sha256,forwards=forwards,exact_baseline_replays=replays,max_relative_edit_roundoff=maximum,path=str(outpath),sha256=sha(outpath))
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
