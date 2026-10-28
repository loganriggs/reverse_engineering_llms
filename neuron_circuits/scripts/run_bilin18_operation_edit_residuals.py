"""H745: exact frozen-subspace residual decomposition; no fitting or model forwards."""
from pathlib import Path
import json,hashlib
from collections import defaultdict
import torch
from run_bilin18_query_key_edit_subspaces import rotate,projected_edit
ROOT=Path('/home/loganriggs/Coding/neuron_circuits');ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM='bilin18_operation_edit_residuals'
def main():
 torch.set_num_threads(6)
 paths=[ART/'bilin18_frozen_operation_generalization_captures.pt',ART/'bilin18_frozen_operation_generalization.json',ART/'bilin18_query_key_edit_subspaces_captures.pt']
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 hashes={str(p):sha(p) for p in paths}
 cfg={'experiment':'H745','inputs':hashes,'design':'No fitting. Exact bilinear expansion of full factor2 donor coefficient minus raw projected edit; separately measure norm-restoration contribution. Single query/key residual rescue is algebraic coefficient diagnostic only. All H744 strata retained; native competence carried through.','whole_model_forwards':0}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cap=torch.load(paths[0],weights_only=False);data=json.loads(paths[1].read_text());bases=torch.load(paths[2],weights_only=False)['bases'];ix={(r['id'],r['receiver']):r for r in data['records'] if r['scope']=='pooled'}
 records=[];errors=[];replay=[];lam=4.1875
 for key,donor in cap['pairs']:
  r,d=cap['states'][key],cap['states'][donor];e=cap['examples'][key];meta=ix[key];pos=[len(e['ids'])-1,e['source_position']]
  raw=[projected_edit(r[u],d[u],bases['pooled',4,None,b],False) for u,b in [('uq2','q2'),('uk2','k2')]]
  norm=[projected_edit(r[u],d[u],bases['pooled',4,None,b],True) for u,b in [('uq2','q2'),('uk2','k2')]]
  q,k=[rotate(x,p) for x,p in zip(raw,pos)];qn,kn=[rotate(x,p) for x,p in zip(norm,pos)];dq,dk=d['q2']-q,d['k2']-k
  gain=lam*(r['q1']@r['k1']/128)/128
  scalar=lambda a,b:float(gain*(a@b))
  native=scalar(r['q2'],r['k2']);target=scalar(d['q2'],d['k2']);raw_alpha=scalar(q,k);norm_alpha=scalar(qn,kn)
  qt,kt,inter=scalar(dq,k),scalar(q,dk),scalar(dq,dk)
  errors.append(abs(target-raw_alpha-qt-kt-inter));replay.append(abs(norm_alpha-meta['predicted_alpha']))
  records.append({**{s:meta[s] for s in ['id','receiver','name_set','wording','format','family','item']},'native':native,'target':target,'raw':raw_alpha,'norm':norm_alpha,'query_residual':qt,'key_residual':kt,'interaction':inter,'norm_restoration':norm_alpha-raw_alpha,'query_rescue_norm':scalar(d['q2'],kn),'key_rescue_norm':scalar(qn,d['k2']),'query_energy':float(((d['uq2']-r['uq2'])@bases['pooled',4,None,'q2']).square().sum()/(d['uq2']-r['uq2']).square().sum()),'key_energy':float(((d['uk2']-r['uk2'])@bases['pooled',4,None,'k2']).square().sum()/(d['uk2']-r['uk2']).square().sum())})
 groups=defaultdict(list)
 for r in records:groups[r['name_set'],r['wording'],r['format']].append(r)
 summary=[]
 for cell,rr in groups.items():
  den=sum((r['target']-r['native'])**2 for r in rr)**.5
  s={'cell':list(cell),'contexts':len(rr),'competent':all(data['competence']['|'.join([*cell,f])]['pass'] for f in ['entity_binding','sequence_copy'])}
  s['relative_error']={k:sum((r[k]-r['target'])**2 for r in rr)**.5/den for k in ['raw','norm','query_rescue_norm','key_rescue_norm']}
  s['component_relative_norm']={k:sum(r[k]**2 for r in rr)**.5/den for k in ['query_residual','key_residual','interaction','norm_restoration']}
  s['mean_edit_energy']={k:sum(r[k] for r in rr)/len(rr) for k in ['query_energy','key_energy']};summary.append(s)
 audits={'hashes_unchanged':all(sha(p)==hashes[str(p)] for p in paths),'expansion_max_abs':max(errors),'H744_replay_max_abs':max(replay),'finite':all(torch.isfinite(torch.tensor([r[k] for k in ['native','target','raw','norm','query_residual','key_residual','interaction']])).all().item() for r in records),'whole_model_forwards':0,'test_evaluations':0}
 assert audits['hashes_unchanged'] and audits['finite'] and max(errors)<1e-10 and max(replay)<1e-10
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'audits':audits},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2))
if __name__=='__main__':main()
