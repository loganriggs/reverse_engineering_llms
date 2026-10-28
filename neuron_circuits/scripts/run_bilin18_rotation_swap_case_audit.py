"""H709: context/prediction audit of exploratory scalar swap results."""
import torch,json,time,hashlib
from pathlib import Path
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_rotation_swap_case_audit'
def main():
 start=time.time();paths={'data':ART/'bilin18_early_bsf_data_captures.pt','results':ART/'bilin18_rotation_context_swap.json','captures':ART/'bilin18_rotation_context_swap_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);j=json.loads(paths['results'].read_text());capture=torch.load(paths['captures'],weights_only=False);tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);cfg=dict(experiment='H709',hashes=hashes,design='Posthoc context/native-prediction and influence audit for every H708coordinate edit; directions121/208 highlighted only after observed results. Group bothdirections ofeachcontextpair. Compare control gains, edit norm relative toreceiver residual, nativeentropy, prefix snippets. No new model executions.',limitations='Posthoc selected cases, not new confirmation; lexical observations generate hypotheses, not validated direction labels. No edited logits saved by H708, so report only nativepredictions and recorded aggregate KL.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[]
 def context(endpoint):
  doc=2048+endpoint//16;pos=int(data['positions'][endpoint%16]);ids=data['all_token_ids'][doc];logits=capture['native_logits'][endpoint].double();p=logits.softmax(-1);top=p.topk(5);return dict(document=doc,position=pos,token=tok.decode([int(ids[pos])]),prefix=tok.decode(ids[max(0,pos-48):pos+1].tolist()),observed_suffix=tok.decode(ids[pos+1:pos+9].tolist()),native_entropy=float(-(p*p.clamp_min(1e-300).log()).sum()),native_top5=[dict(token=tok.decode([int(t)]),probability=float(v)) for v,t in zip(top.values,top.indices)])
 for r in j['records']:
  if r['mode']!='coordinate':continue
  counterparts={v['mode']:v for v in j['records'] if v['direction']==r['direction'] and v['receiver']==r['receiver'] and v['donor']==r['donor']};receiver=context(r['receiver']);donor=context(r['donor']);raw=data['activations']['residual1'][2048:2304].reshape(-1,1152)[r['receiver']];records.append(dict(direction=r['direction'],receiver_index=r['receiver'],donor_index=r['donor'],receiver=receiver,donor=donor,coordinate_relative_edit_norm=r['coordinate_norm']/float(raw.norm()),gains={m:v['donor_kl_gain'] for m,v in counterparts.items()},receiver_kls={m:v['receiver_kl'] for m,v in counterparts.items()},donor_kl_baseline=r['donor_kl_baseline']))
 summaries=[]
 for f in sorted({r['direction'] for r in records}):
  rr=[r for r in records if r['direction']==f];groups={tuple(sorted((r['receiver_index'],r['donor_index']))) for r in rr};pairrows=[]
  for group in sorted(groups):
   vv=[r for r in rr if tuple(sorted((r['receiver_index'],r['donor_index'])))==group];pairrows.append(dict(pair=group,mean_coordinate_gain=sum(r['gains']['coordinate'] for r in vv)/len(vv),mean_pca_gain=sum(r['gains']['pca'] for r in vv)/len(vv),mean_random_gain=sum(r['gains']['random'] for r in vv)/len(vv)))
  vals=[r['mean_coordinate_gain'] for r in pairrows];summaries.append(dict(direction=f,pairs=pairrows,mean=sum(vals)/len(vals),leave_one_pair_out_means=[(sum(vals)-v)/(len(vals)-1) for v in vals]))
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},records=len(records),whole_model_forwards=0,test_evaluations=0);assert all(audits['hashes_unchanged'].values());(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,summaries=summaries,audits=audits,elapsed_seconds=time.time()-start),indent=2))
 for f in [121,208]:
  print('SUMMARY',next(r for r in summaries if r['direction']==f))
  for r in records:
   if r['direction']==f:print(json.dumps(r),flush=True)
if __name__=='__main__':main()
