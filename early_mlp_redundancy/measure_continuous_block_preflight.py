"""M111 frozen-state preflight: continuity at all117 known boundaries and amplitude cost.
No training, task labels, or native forward evaluations. This is not a fair trained comparison.
"""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from continuous_block_shrinkage import block_shrinkage


def main():
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    check=json.loads((V/'continuous_block_shrinkage_checks.json').read_text());assert check['status']=='passed' and check['implementation_sha256']==sha(V.parent/'continuous_block_shrinkage.py')
    p=json.loads((V/'gate_boundary_protocol.json').read_text());nr=json.loads((V/'gate_boundary_native.json').read_text());assert sha(nr['path'])==nr['sha256'];saved=torch.load(nr['path'],weights_only=True)
    files=json.loads((V/'block_change_roles_whole.json').read_text())['files'];cache={}
    for f in files:
      assert sha(f['path'])==f['sha256'];z=torch.load(f['path'],weights_only=True);cache[f['pair_id']]=torch.stack([z['writes'][i][0,-1] for i in [0,1]]).double()
    states={}
    for e in p['models']:
      assert sha(e['path'])==e['sha256'];z=torch.load(e['path'],weights_only=True);states[e['seed'],e['arm']]=z
    rows=[];crossings=[];maximum=0.
    for row in nr['rows']:
      z=states[row['seed'],row['arm']];st={k:v.double() for k,v in z['state'].items()};mu=z['mean'].double();scale=z['scale'].double();D=torch.nn.functional.normalize(st['dictionary'],dim=0);w=cache[row['pair_id']];raw=((w-mu)/scale)@st['encoder'].T+st['encoder_bias'];blocks=raw.reshape(2,32,4)
      energy=blocks.square().sum(-1);gates=torch.zeros_like(energy).scatter_(-1,energy.topk(8,-1).indices,1);hard=blocks*gates[:,:,None];soft=block_shrinkage(blocks,8)
      # Independent NumPy sorted-norm formula for every endpoint.
      a=blocks.numpy();norm=np.linalg.norm(a,axis=-1);tau=np.sort(norm,axis=-1)[:,-9,None];ref=a*(np.maximum(norm-tau,0)/np.where(norm>0,norm,1))[...,None];err=float(np.max(np.abs(ref-soft.numpy())));maximum=max(maximum,err);assert err<1e-10
      hr=hard.reshape(2,128)@D.T*scale+st['bias']*scale+mu;sr=soft.reshape(2,128)@D.T*scale+st['bias']*scale+mu
      targetdelta=w[1]-w[0];norms=blocks.norm(dim=-1);threshold=norms.topk(9,-1).values[:,-1]
      rows.append({k:row[k] for k in ['pair_id','family','seed','arm']}|dict(hard_error=float((hr-w).square().sum()),soft_error=float((sr-w).square().sum()),target_energy=float(w.square().sum()),hard_delta_error=float((hr[1]-hr[0]-targetdelta).square().sum()),soft_delta_error=float((sr[1]-sr[0]-targetdelta).square().sum()),target_delta_energy=float(targetdelta.square().sum()),hard_code_energy=float(hard.square().sum()),soft_code_energy=float(soft.square().sum()),soft_active_blocks=(soft.norm(dim=-1)>0).sum(-1).tolist(),thresholds=threshold.tolist()))
      if row['status']!='evaluated':continue
      key=f"{row['pair_id']}|{row['seed']}|{row['arm']}";record=saved[key];a,b=record['a'],record['b'];t=row['first_boundary'];measure=[]
      for old in row['finite_limits']:
        h=old['h'];left=block_shrinkage(a+(t-h)*b,8).reshape(128)@D.T*scale;right=block_shrinkage(a+(t+h)*b,8).reshape(128)@D.T*scale;diff=float((right-left).norm())
        bound=(1+np.sqrt(32))*2*h*float(b.norm())*float(torch.linalg.matrix_norm(D,ord=2))*float(scale)
        assert diff<=bound+1e-9
        measure.append(dict(h=h,soft_change_norm=diff,hard_change_norm=old['decoded_change_norm'],native_change_norm=old['native_change_norm'],proven_lipschitz_bound=bound))
      crossings.append(dict(key=key,finite_limits=measure))
    pooled=[]
    for e in p['models']:
      ss=[x for x in rows if x['seed']==e['seed'] and x['arm']==e['arm']];summ=lambda k:sum(x[k] for x in ss)
      pooled.append(dict(seed=e['seed'],arm=e['arm'],soft_over_hard_code_energy=summ('soft_code_energy')/summ('hard_code_energy'),hard_relative_write_error=summ('hard_error')/summ('target_energy'),soft_relative_write_error=summ('soft_error')/summ('target_energy'),hard_relative_change_error=summ('hard_delta_error')/summ('target_delta_energy'),soft_relative_change_error=summ('soft_delta_error')/summ('target_delta_energy')))
    assert len(rows)==224 and len(crossings)==117
    out=dict(status='passed',source_sha256=sha(__file__),implementation_sha256=sha(V.parent/'continuous_block_shrinkage.py'),boundary_native_sha256=sha(V/'gate_boundary_native.json'),max_numpy_error=maximum,rows=rows,crossings=crossings,pooled=pooled,scope='Cold map substitution in frozen hard-gate-trained models. No training or downstream fidelity evaluation; cannot establish that a trained continuous featurizer is better or worse.')
    dest=V/'continuous_block_preflight.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('CASES',len(rows),'CROSSINGS',len(crossings),'MAX_NUMPY',maximum)
    for x in pooled:print(x)
if __name__=='__main__':main()
