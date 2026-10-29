"""M48 all-group exclusive/shared union audit on natural evaluation writes."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'shared_group_decomposition_protocol.json';q=json.loads(pp.read_text());assert q['source_sha256']==sha(Path(__file__));mp=V/'coactivation_groups.json';assert sha(mp)==q['models_sha256'];manifest=json.loads(mp.read_text());assert json.loads((V/'coactivation_groups_audit.json').read_text())['result_sha256']==sha(mp);m4=json.loads((V/'natural_token_context.json').read_text());cache=Path(m4['cache']);assert sha(cache)==m4['cache_sha256'];data=torch.load(cache,weights_only=True);Y=data['Y'][1][4096:8192].cuda();records=[];checks=[];samples=torch.tensor([0,1,127,128,1023,2048,3072,4095],device=Y.device)
 for e in [x for x in manifest['models'] if x['kind']=='shared_fixed']:
  path=Path(e['path']);assert sha(path)==e['sha256'];state=torch.load(path,weights_only=True);f=Featurizer('shared_fixed',e['seed']).cuda();f.load_state_dict(state['state']);mu=state['mean'].cuda();scale=state['scale'].cuda();x=(Y-mu)/scale;a=torch.nn.functional.linear(x,f.encoder,f.encoder_bias);scores=a.square()@f.membership.T;gates=torch.zeros_like(scores).scatter_(-1,scores.topk(4,dim=-1).indices,1);counts=gates@f.membership;support=counts>0;pred,qf,code=f(x);assert torch.equal(qf.bool(),support);assert torch.equal(code,a*support);D=torch.nn.functional.normalize(f.dictionary,dim=0);M=f.membership.bool();assert bool((M.sum(0)==2).all()) and bool((M.sum(1)==8).all());maxerr=0.;max_numpy=0.
  # Independent set-based audit of all gates on eight fixed positions per candidate.
  mm=M.cpu().numpy();aa=a[samples].cpu().double().numpy();dd=D.cpu().double().numpy();gg=gates[samples].cpu().numpy()
  for group in range(32):
   chosen=gates[:,group,None].bool()&M[group];exclusive=chosen&(counts==1);shared=chosen&(counts>1);remaining=gates.clone();remaining[:,group]=0;after=(remaining@f.membership)>0
   assert torch.equal(support&~after,exclusive);assert torch.equal(chosen,exclusive|shared);assert not bool((exclusive&shared).any());assert bool((after[shared]).all())
   de=(a*exclusive)@D.T*scale;ds=(a*shared)@D.T*scale;du=(a*chosen)@D.T*scale;err=float((de.double()+ds.double()-du.double()).norm()/du.double().norm().clamp_min(1));assert err<1e-5;maxerr=max(maxerr,err)
   for j in range(len(samples)):
    active=set(np.flatnonzero(gg[j]));before=set().union(*(set(np.flatnonzero(mm[k])) for k in active));others=active-{group};afterset=set().union(*(set(np.flatnonzero(mm[k])) for k in others));selected=set(np.flatnonzero(mm[group])) if group in active else set();ex=selected-afterset;sh=selected&afterset;assert before-afterset==ex
    for mask,expected,delta in [(exclusive,ex,de),(shared,sh,ds),(chosen,selected,du)]:
     assert set(torch.where(mask[samples[j]])[0].cpu().tolist())==expected;coeff=np.zeros(128);ii=list(expected);coeff[ii]=aa[j,ii];replayed=coeff@dd.T*float(scale);actual=delta[samples[j]].cpu().double().numpy();rel=np.linalg.norm(replayed-actual)/max(np.linalg.norm(actual),1);assert rel<1e-5;max_numpy=max(max_numpy,float(rel))
   row={k:e[k] for k in ['seed','source_arm','grouping']};row.update(group=group,positions=4096,gate_active_positions=int(gates[:,group].sum()),selected_atom_positions=int(chosen.sum()),exclusive_atom_positions=int(exclusive.sum()),shared_atom_positions=int(shared.sum()),exclusive_energy=float(de.double().square().sum()),shared_energy=float(ds.double().square().sum()),union_energy=float(du.double().square().sum()),exclusive_shared_inner_product=float((de.double()*ds.double()).sum()),zero_exclusive_positions=int(((exclusive.sum(-1)==0)&gates[:,group].bool()).sum()));records.append(row)
  checks.append({k:e[k] for k in ['seed','source_arm','grouping']}|dict(all_positions=4096,groups=32,max_union_relative_error=maxerr,max_numpy_decode_relative_error=max_numpy,independent_set_cases=32*len(samples)))
  print('CHECKED',e['seed'],e['arm'],flush=True)
 summaries=[]
 for check in checks:
  rr=[x for x in records if all(x[k]==check[k] for k in ['seed','source_arm','grouping'])];tot={k:sum(x[k] for x in rr) for k in ['gate_active_positions','selected_atom_positions','exclusive_atom_positions','shared_atom_positions','exclusive_energy','shared_energy','union_energy','exclusive_shared_inner_product','zero_exclusive_positions']};assert tot['gate_active_positions']==4096*4;assert tot['selected_atom_positions']==4096*32;assert tot['selected_atom_positions']==tot['exclusive_atom_positions']+tot['shared_atom_positions'];summaries.append({k:check[k] for k in ['seed','source_arm','grouping']}|tot|dict(shared_membership_fraction=tot['shared_atom_positions']/tot['selected_atom_positions']))
 result=dict(status='passed',protocol_sha256=sha(pp),models_sha256=sha(mp),native_write_cache_sha256=sha(cache),checks=checks,records=records,summaries=summaries,scope='All eight frozen shared candidates, all32groups,4096natural evaluation positions each. Exact gate/atom set identities and independent sampled NumPy decodes; this audits architectural overlap, not semantic/native reuse. Group-summed shared counts count a doubly selected atom twice; not unique atom counts.')
 op=V/'shared_group_decomposition.json';assert not op.exists();op.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
