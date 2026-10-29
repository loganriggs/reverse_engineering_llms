"""M75 native holdout inventory and all saved vectors."""
from pathlib import Path
import hashlib,json
import torch,numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(x):
 x=np.asarray(x,dtype=np.float64);x=x-x.max();return x-np.log(np.exp(x).sum())
def margin(x,g,v):
 x=np.asarray(x,dtype=np.float64)
 def lse(a):return a.max()+np.log(np.exp(a-a.max()).sum())
 return float(lse(x[g[v]])-lse(x[g[1-v]]))
def main():
 torch.set_num_threads(1);rp=V/'whole_task_native.json';r=json.loads(rp.read_text());pp=V/'whole_task_native_protocol.json';q=json.loads(pp.read_text());assert r['protocol_sha256']==sha(pp);assert r['source_sha256']==sha(P/'probe_whole_task_native.py');sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];s=torch.load(sp,weights_only=True);spl=json.loads((V/'diverse_supervision_split.json').read_text());assert not set(spl['train_pair_ids'])&{p['id'] for p in q['pairs']};assert set(q['training_exclusions'])=={p['family'] for p in q['pairs']};maxerr=0
 def check(a,b):
  nonlocal maxerr
  assert np.isclose(a,b,rtol=1e-8,atol=1e-9),(a,b);maxerr=max(maxerr,abs(a-b))
 keys=['pair_id','endpoint','layer','region'];assign={tuple(a[k] for k in keys):a for a in q['assignments']};assert len(assign)==len(r['records'])==336
 for row in r['records']:
  a=assign[tuple(row[k] for k in keys)];assert all(row[k]==a[k] for k in a);p=next(p for p in q['pairs'] if p['id']==row['pair_id']);x,y=p['token_ids'];changed=[i for i,(u,v) in enumerate(zip(x,y)) if u!=v] if len(x)==len(y) else [];assert changed==p['changed_positions']
  if not a['eligible']:assert row['status']=='ineligible' and len(x)!=len(y);continue
  ep=row['endpoint'];region=row['region'];pos=a['recipient_positions'];assert pos==(changed if region=='changed_input_positions' else [len(p['token_ids'][ep])-1] if region=='final_position' else []);key=p['id']+f'|{ep}';b=s['baselines'][key];z=b['logits'].numpy();o=s['changed'][row['sample_key']].numpy();delta=s['deltas'][row['sample_key']];ref=delta['donor'].numpy()-delta['recipient'].numpy();assert np.array_equal(ref,delta['delta'].numpy());check(float(np.sum(ref.astype(np.float64)**2)),row['delta_energy']);blp=lp(z);olp=lp(o);check(float(np.sum(np.exp(blp)*(olp-blp)**2)),row['response_energy']);check(float(np.sum(np.exp(blp)*(blp-olp))),row['output_KL']);bm=margin(z,b['groups'],b['variant']);check(bm,row['native_margin']);check(margin(o,b['groups'],b['variant'])-bm,row['margin_change']);assert row['native_binary']==(bm>0);assert row['native_top1']==(int(z.argmax()) in b['groups'][b['variant']]);assert row['changed_top1']==(int(o.argmax()) in b['groups'][b['variant']])
  if region=='identity':assert row['noop'] and row['delta_energy']==0 and row['response_energy']==0 and row['margin_change']==0
 for summary in r['summaries']:
  allrows=[x for x in r['records'] if all(x[k]==summary[k] for k in ['family','layer','region'])];rr=[x for x in allrows if x['status']=='evaluated'];assert summary['assigned']==len(allrows) and summary['evaluated']==len(rr);assert summary['ineligible']==len(allrows)-len(rr);check(sum(x['response_energy'] for x in rr),summary['response_energy']);check(sum(x['delta_energy'] for x in rr),summary['delta_energy']);check(sum(x['output_KL'] for x in rr)/len(rr),summary['mean_KL']);check(sum(abs(x['margin_change']) for x in rr)/len(rr),summary['mean_absolute_margin_change']);assert summary['margin_reductions']==sum(x['margin_change']<0 for x in rr)
 assert r['forward_count']==112+sum(x['status']=='evaluated' and not x['noop'] for x in r['records']);assert len(r['summaries'])==21
 replays=0
 out=dict(status='passed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),assignments=336,evaluated_vectors=328,ineligible=8,summaries=21,max_absolute_numeric_error=maxerr,color_previous_replays=replays,scope='All saved final vectors independently NumPy rescored; address-based alignment/training-family exclusions/identity/common native delta formula and aggregates verified. Transformer not independently reimplemented.')
 op=V/'whole_task_native_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
