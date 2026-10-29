"""Independently audit cubic edits; summarize all native causal outcomes."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha


def main():
    torch.set_num_threads(1)
    p=json.loads((V/'cube_causal_protocol.json').read_text())
    receipt=json.loads((V/'cube_causal_native.json').read_text())
    assert receipt['protocol_sha256']==sha(V/'cube_causal_protocol.json')
    assert receipt['producer_sha256']==sha(V.parent/'probe_cube_causal.py')
    assert sha(receipt['path'])==receipt['sha256']
    saved=torch.load(receipt['path'],weights_only=True)
    nr=json.loads((V/'repetition_cube_native.json').read_text());assert sha(nr['states_path'])==nr['states_sha256']
    native=torch.load(nr['states_path'],weights_only=True)
    rows=json.loads((V/'repetition_cube_protocol.json').read_text())['rows']
    # Build design independently of production helper, then use general solve.
    design=np.array([[np.prod([2*((i>>j)&1)-1 for j in range(3) if mask&(1<<j)]) for mask in range(8)] for i in range(8)],dtype=float)
    arms=['remove_cubic']+['scramble_'+str(s) for s in p['scramble_seeds']]
    expected=set();summary=[];max_error=0.;checks=0
    for case in range(4):
      for fmt in ['semicolon','comma']:
        items=sorted([r for r in rows if r.get('case')==case and r.get('format')==fmt],key=lambda r:r['vertex'])
        assert len(items)==8
        ref=np.stack([native[r['id']]['logits'].double().numpy() for r in items])
        tokens=items[0]['canonical_answer_tokens'];margin=ref[:,tokens[1]]-ref[:,tokens[0]]
        refc=np.linalg.solve(design,margin)
        for layer in p['layers']:
          writes=np.stack([native[r['id']]['layers'][layer]['write'][0,-1].double().numpy() for r in items])
          coeff=np.linalg.solve(design,writes);c=coeff[7]
          assert np.allclose(design@coeff,writes,rtol=1e-12,atol=1e-10)
          for arm in arms:
            target=c.copy()
            if arm!='remove_cubic':
              rng=np.random.default_rng(int(arm.split('_')[1])+100*layer)
              perm=rng.permutation(c.size);sign=rng.choice([-1,1],c.size);target=c[perm]*sign
            assert np.allclose(saved['directions'][f'{case}|{fmt}|{layer}'][arm].numpy(),target,rtol=1e-10,atol=1e-10)
            changed=[]
            for r in items:
              key=f"{r['id']}|{layer}|{arm}";expected.add(key);record=saved['results'][key]
              expected_delta=-design[r['vertex'],7]*target
              err=np.linalg.norm(record['effective_delta'].double().numpy()-expected_delta)/max(np.linalg.norm(expected_delta),1e-12)
              max_error=max(max_error,float(err));assert err<1e-4;checks+=1
              changed.append(record['logits'].double().numpy())
            changed=np.stack(changed);cm=changed[:,tokens[1]]-changed[:,tokens[0]];cc=np.linalg.solve(design,cm)
            def logp(x):
              x=x-x.max(axis=-1,keepdims=True)
              return x-np.log(np.exp(x).sum(axis=-1,keepdims=True))
            lr=logp(ref);lc=logp(changed);kl=(np.exp(lr)*(lr-lc)).sum(-1)
            assert np.isfinite(kl).all() and kl.min()>-1e-10
            summary.append(dict(case=case,format=fmt,layer=layer,arm=arm,reference_coefficients=refc.tolist(),edited_coefficients=cc.tolist(),coefficient_changes=(cc-refc).tolist(),mean_KL=float(kl.mean()),per_vertex_KL=kl.tolist(),top1_changes=int(np.sum(ref.argmax(1)!=changed.argmax(1))),candidate_changes=int(np.sum((margin>0)!=(cm>0))),margin_change_rms=float(np.sqrt(np.mean((cm-margin)**2))),cubic_delta=float(cc[7]-refc[7]),direction_norm=float(np.linalg.norm(target))))
    assert set(saved['results'])==expected and len(expected)==768
    assert receipt['forwards']==832 and receipt['exact_baseline_replays']==64
    pooled=[]
    for layer in p['layers']:
      for arm in arms:
        ss=[r for r in summary if r['layer']==layer and r['arm']==arm]
        pooled.append(dict(layer=layer,arm=arm,mean_KL=float(np.mean([r['mean_KL'] for r in ss])),top1_changes=sum(r['top1_changes'] for r in ss),candidate_changes=sum(r['candidate_changes'] for r in ss),margin_change_rms=float(np.sqrt(np.mean([r['margin_change_rms']**2 for r in ss]))),mean_cubic_delta=float(np.mean([r['cubic_delta'] for r in ss])),reference_cubic_energy=sum(r['reference_coefficients'][7]**2 for r in ss),edited_cubic_energy=sum(r['edited_coefficients'][7]**2 for r in ss),coefficient_change_energy=np.square([r['coefficient_changes'] for r in ss]).sum(0).tolist()))
    result=dict(status='audited',source_sha256=sha(__file__),native_receipt_sha256=sha(V/'cube_causal_native.json'),audit=dict(independent_linear_solve=True,edits_checked=checks,exact_key_coverage=True,max_relative_edit_roundoff=max_error),cubes=summary,pooled=pooled)
    dest=V/'cube_causal_analysis.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
    for r in pooled:print({k:v for k,v in r.items() if k!='coefficient_change_energy'})
if __name__=='__main__':main()
