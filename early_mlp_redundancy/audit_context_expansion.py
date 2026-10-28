"""M17 independent token/split and NumPy native competence audit."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch,tiktoken
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lse(x):
 m=x.max();return float(m+np.log(np.exp(x-m).sum()))
pp=P/'context_expansion_protocol.json';p=json.loads(pp.read_text());bp=P/'training_source_context_audit.json';sp=P/'diverse_supervision_split.json';assert sha(bp)==p['bank_sha256'] and sha(sp)==p['lexical_split_sha256'];bank={x['id']:x for x in json.loads(bp.read_text())['rows']};split=json.loads(sp.read_text());sources={k:set() for k in ['train','evaluation']}
for k,ids in [('train',split['train_pair_ids']),('evaluation',split['evaluation_pair_ids'])]:
 for pid in ids:
  a,b=bank[pid]['token_ids'];assert len(a)==len(b)
  for x,y in zip(a,b):
   if x!=y:sources[k].update([x,y])
enc=tiktoken.get_encoding('gpt2');templates={x['id']:x for x in p['templates']};assert len(templates)==6;assert sum(t['split']=='train' for t in templates.values())==4;seen=set();quadrants={}
for row in p['rows']:
 assert row['id'] not in seen;seen.add(row['id']);b=bank[row['base_id']];t=templates[row['template']];ls='train' if row['base_id'] in split['train_pair_ids'] else 'evaluation';assert row['lexical_split']==ls;assert row['template_split']==t['split'];assert row['training_eligible']==(ls=='train' and t['split']=='train');pre=enc.encode(t['prefix']);assert row['prefix_length']==len(pre);assert row['answer_groups']==b['answer_groups'] and row['variants']==b['variants'];q=ls+':'+t['split'];quadrants[q]=quadrants.get(q,0)+1
 for ep in [0,1]:
  assert row['texts'][ep]==t['prefix']+b['texts'][ep];assert row['token_ids'][ep]==enc.encode(row['texts'][ep])==pre+b['token_ids'][ep];assert not set(row['token_ids'][ep])&sources['evaluation' if ls=='train' else 'train']
assert quadrants=={'train:train':264,'evaluation:train':56,'train:evaluation':132,'evaluation:evaluation':28}
ip=P/'context_expansion_competence.json';r=json.loads(ip.read_text());assert sha(pp)==r['protocol_sha256'];ap=Path(r['samples_path']);assert sha(ap)==r['samples_sha256'];samples=torch.load(ap,weights_only=True);rows={x['id']:x for x in p['rows']};results={x['id']:x for x in r['results']};errors=[];keys=set()
for sample in samples:
 pid=sample['id'];ep=sample['endpoint'];assert (pid,ep) not in keys;keys.add((pid,ep));row=rows[pid];x=sample['logits'].numpy().astype(float);g=row['answer_groups'][ep];v=row['variants'][ep];margin=lse(x[g[v]])-lse(x[g[1-v]]);out=results[pid]['endpoints'][ep];errors.append(abs(margin-out['margin']));assert errors[-1]<1e-10;assert out['binary']==(margin>0);assert out['top1']==(int(x.argmax()) in g[v])
assert len(keys)==960
for c in r['cohorts']:
 rr=[x for x in r['results'] if all(x[k]==c[k] for k in ['template','lexical_split','family'])];assert len(rr)==c['pairs'];binary=sum(all(e['binary'] for e in x['endpoints']) for x in rr)/len(rr);top=sum(e['top1'] for x in rr for e in x['endpoints'])/(2*len(rr));assert binary==c['both_binary_rate'] and top==c['top1_rate'];assert c['pass_gate']==(binary>=.75 and top>=.5)
out=dict(status='passed',pairs=len(rows),endpoints=len(keys),quadrant_pairs=quadrants,source_to_opposite_prompt_intersections=0,exact_token_boundary_checks=960,native_logit_checks=960,cohort_gate_checks=len(r['cohorts']),max_margin_error=max(errors),protocol_sha256=sha(pp),competence_sha256=sha(ip));(P/'context_expansion_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
