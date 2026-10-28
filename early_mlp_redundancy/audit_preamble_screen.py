"""Independent saved-logit and membership audit for EMLP-P8."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import tiktoken
P=Path(__file__).resolve().parent

def lse(x):
 m=np.max(x);return m+np.log(np.exp(x-m).sum())
def margin(x,g,v):return lse(x[g[v]])-lse(x[g[1-v]])
def kl(a,b):
 a=a-lse(a);b=b-lse(b);return np.sum(np.exp(a)*(a-b))
def main():
 cpath=P/'validation/preamble_competence.json';tpath=P/'validation/preamble_transfer.json'
 c=json.loads(cpath.read_text());t=json.loads(tpath.read_text());source=json.loads((P/'validation/training_source_context_audit.json').read_text())
 old={p['id']:p for p in source['rows']};assert len(old)==80
 selected={p['id'] for p in source['rows'] if p['family'] in c['families']};assert len(selected)==36
 assert len(c['rows'])==72 and c['forward_count']==144
 tok=tiktoken.get_encoding('gpt2')
 for pre in range(2):assert {r['id'] for r in c['rows'] if r['preamble_index']==pre}==selected
 for r in c['rows']:
  for i in (0,1):
   assert r['texts'][i]==c['preambles'][r['preamble_index']]+old[r['id']]['texts'][i]
   assert tok.encode(r['texts'][i])==r['token_ids'][i]
 for cohort in c['cohorts']:
  rr=[r for r in c['rows'] if r['family']==cohort['family'] and r['preamble_index']==cohort['preamble_index']]
  binary=sum(r['both_binary'] for r in rr)/len(rr);top=sum(r['top1_rate'] for r in rr)/len(rr)
  assert binary==cohort['both_endpoint_binary_rate'] and top==cohort['full_vocab_top1_rate']
  assert cohort['competence_gate_pass']==(binary>=.75 and top>=.5)
 passed={(x['family'],x['preamble_index']) for x in c['cohorts'] if x['competence_gate_pass'] and x['all_single_token']}
 eligible={(r['id'],r['preamble_index']) for r in c['rows'] if (r['family'],r['preamble_index']) in passed}
 assert len(eligible)==48
 assert len(t['rows'])==48*2*7 and t['forward_count']==48*22
 for ident,pre in eligible:
  for rec in (0,1):
   rr=[r for r in t['rows'] if (r['id'],r['preamble_index'],r['receiver'])==(ident,pre,rec)]
   assert sorted(r['kind'] for r in rr)==sorted(['context_specific','original_context','original_norm_matched']+['random']*4)
 errors=[]
 for x in torch.load('/tmp/emlp_p8_competence_logits.pt',weights_only=True):
  r=c['rows'][x['row']]['endpoints'][x['endpoint']];lg=x['logits'].double().numpy();v=x['variant'];g=x['groups']
  errors.append(abs(margin(lg,g,v)-r['margin']));assert int(lg.argmax())==r['top_token'];assert r['binary']==(margin(lg,g,v)>0)
 samples=torch.load('/tmp/emlp_p8_transfer_samples.pt',weights_only=True)
 for x in samples['samples']:
  r=t['rows'][x['row']];a=x['teacher'].double().numpy();b=x['baseline'].double().numpy();z=x['candidate'].double().numpy()
  errors.extend([abs(kl(a,b)-r['native_donor_KL']),abs(kl(a,z)-r['remaining_donor_KL']),abs(margin(z,x['groups'],x['variant'])-margin(b,x['groups'],x['variant'])-r['margin_gain'])])
 for v in samples['vectors']:
  a=v['old'].double().numpy().reshape(-1);b=v['new'].double().numpy().reshape(-1)
  rr=[r for r in t['rows'] if (r['id'],r['preamble_index'],r['receiver'])==(v['id'],v['preamble_index'],v['receiver'])]
  assert len(rr)==7
  cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
  error=float(np.square(a-b).sum()/np.square(b).sum())
  for r in rr:
   assert abs(cosine-r['response_cosine'])<1e-6
   assert abs(error-r['relative_vector_error'])<1e-5*max(1,error)
 for co in t['cohorts']:
  for kind,met in co['responses'].items():
   rr=[r for r in t['rows'] if (r['family'],r['preamble_index'],r['kind'])==(co['family'],co['preamble_index'],kind)]
   assert len(rr)==met['n'];assert abs(sum(r['remaining_donor_KL'] for r in rr)/sum(r['native_donor_KL'] for r in rr)-met['remaining_donor_KL_ratio'])<1e-12
 assert max(errors)<1e-10
 result=dict(status='passed',competence_rows=72,transfer_pairs=48,competence_logit_checks=4,transfer_logit_checks=len(samples['samples']),max_absolute_metric_error=max(errors),competence_sha256=hashlib.sha256(cpath.read_bytes()).hexdigest(),transfer_sha256=hashlib.sha256(tpath.read_bytes()).hexdigest())
 (P/'validation/preamble_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
