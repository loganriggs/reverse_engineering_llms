"""M109 NumPy reconstruction audit and complete factor-response comparison."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha


def main():
    torch.set_num_threads(1)
    pp=V/'cube_featurizer_protocol.json';p=json.loads(pp.read_text())
    rp=V/'cube_featurizer_native.json';r=json.loads(rp.read_text())
    assert r['protocol_sha256']==sha(pp) and r['source_sha256']==sha(V.parent/'probe_cube_featurizers.py')
    assert sha(r['path'])==r['sha256'];saved=torch.load(r['path'],weights_only=True)
    nr=json.loads((V/'repetition_cube_native.json').read_text());assert sha(nr['states_path'])==nr['states_sha256'];native=torch.load(nr['states_path'],weights_only=True)
    rows=json.loads((V/'repetition_cube_protocol.json').read_text())['rows'];rows=[x for x in rows if x['cohort']!='historical_replay']
    design=np.array([[np.prod([2*((i>>j)&1)-1 for j in range(3) if mask&(1<<j)]) for mask in range(8)] for i in range(8)],float)
    summary=[];expected=set();maxerr=0.;checks=0
    for e in p['models']:
      assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True);state={k:t.double().numpy() for k,t in s['state'].items()};mu=s['mean'].double().numpy();scale=s['scale'].double().numpy()
      D=state['dictionary'];D=D/np.linalg.norm(D,axis=0)
      for case in range(4):
       for fmt in ['semicolon','comma']:
        items=sorted([x for x in rows if x['case']==case and x['format']==fmt],key=lambda x:x['vertex'])
        w=np.stack([native[x['id']]['layers'][1]['write'][0,-1].double().numpy() for x in items]);wc=np.linalg.solve(design,w)
        z=np.stack([native[x['id']]['logits'].double().numpy() for x in items]);tokens=items[0]['canonical_answer_tokens'];margin=z[:,tokens[1]]-z[:,tokens[0]];mc=np.linalg.solve(design,margin)
        x=(w-mu)/scale;a=x@state['encoder'].T+state['encoder_bias'];energy=a*a@state['membership'].T
        selected=np.argsort(energy,axis=1)[:,-8:];gates=np.zeros_like(energy);np.put_along_axis(gates,selected,1,axis=1);q=(gates@state['membership'])>0
        assert (q.sum(1)==32).all()
        sparse=((a*q)@D.T+state['bias'])*scale+mu
        dense=(np.linalg.lstsq(D,(x-state['bias']).T,rcond=None)[0].T@D.T+state['bias'])*scale+mu
        for mode,target in [('sparse',sparse),('dense',dense)]:
          records=[]
          for j,item in enumerate(items):
            key=f"{item['id']}|{e['seed']}|{e['arm']}|{mode}";expected.add(key);record=saved[key];records.append(record)
            err=np.linalg.norm(record['write'].double().numpy()-target[j])/max(np.linalg.norm(target[j]),1)
            maxerr=max(maxerr,float(err));assert err<1e-5;checks+=1
          rw=np.stack([s['write'].double().numpy() for s in records]);rc=np.linalg.solve(design,rw)
          rz=np.stack([s['logits'].double().numpy() for s in records]);rm=rz[:,tokens[1]]-rz[:,tokens[0]];rmc=np.linalg.solve(design,rm)
          def logp(t):
            t=t-t.max(1,keepdims=True);return t-np.log(np.exp(t).sum(1,keepdims=True))
          lp=logp(z);rp_=logp(rz);kl=(np.exp(lp)*(lp-rp_)).sum(1);assert kl.min()>-1e-10
          def by_order(t):return [float(np.square(t[[k for k in range(8) if k.bit_count()==d]]).sum()) for d in range(4)]
          summary.append(dict(seed=e['seed'],arm=e['arm'],mode=mode,case=case,format=fmt,mean_KL=float(kl.mean()),top1_changes=int(np.sum(z.argmax(1)!=rz.argmax(1))),candidate_changes=int(np.sum((margin>0)!=(rm>0))),native_margin_coefficients=mc.tolist(),edited_margin_coefficients=rmc.tolist(),margin_coefficient_error_by_order=by_order(rmc-mc),native_margin_energy_by_order=by_order(mc),local_coefficient_error_by_order=by_order(rc-wc),native_local_energy_by_order=by_order(wc),reconstructed_local_energy_by_order=by_order(rc)))
    assert set(saved)==expected and len(expected)==512
    assert r['forward_count']==576 and r['baseline_replays']==64
    pooled=[]
    for e in p['models']:
      for mode in ['sparse','dense']:
        ss=[x for x in summary if x['seed']==e['seed'] and x['arm']==e['arm'] and x['mode']==mode]
        sums={k:np.sum([x[k] for x in ss],axis=0) for k in ['margin_coefficient_error_by_order','native_margin_energy_by_order','local_coefficient_error_by_order','native_local_energy_by_order','reconstructed_local_energy_by_order']}
        pooled.append(dict(seed=e['seed'],arm=e['arm'],mode=mode,mean_KL=float(np.mean([x['mean_KL'] for x in ss])),top1_changes=sum(x['top1_changes'] for x in ss),candidate_changes=sum(x['candidate_changes'] for x in ss),margin_relative_error_by_order=(sums['margin_coefficient_error_by_order']/sums['native_margin_energy_by_order']).tolist(),local_relative_error_by_order=(sums['local_coefficient_error_by_order']/sums['native_local_energy_by_order']).tolist(),margin_variation_relative_error=float(sums['margin_coefficient_error_by_order'][1:].sum()/sums['native_margin_energy_by_order'][1:].sum()),local_variation_relative_error=float(sums['local_coefficient_error_by_order'][1:].sum()/sums['native_local_energy_by_order'][1:].sum())))
    out=dict(status='audited',source_sha256=sha(__file__),native_sha256=sha(V/'cube_featurizer_native.json'),audit=dict(numpy_decodes_checked=checks,max_relative_decode_error=maxerr,complete_key_coverage=True),cubes=summary,pooled=pooled)
    dest=V/'cube_featurizer_analysis.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
    print('AUDIT',out['audit'])
    for x in pooled:print(x)
if __name__=='__main__':main()
