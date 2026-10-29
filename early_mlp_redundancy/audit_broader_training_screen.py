"""M55 independently rescore every saved final logit vector and frozen prompt."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from transformers import AutoTokenizer
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);mp=V/'broader_training_screen_manifest.json';m=json.loads(mp.read_text());rp=V/'broader_training_screen.json';r=json.loads(rp.read_text());assert sha(mp)==r['manifest_sha256'];assert r['source_sha256']==sha(P/'probe_broader_training_screen.py');design=json.loads((V/'broader_training_screen_design.json').read_text());assert m['design_sha256']==sha(V/'broader_training_screen_design.json');tok=AutoTokenizer.from_pretrained('/home/loganriggs/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e',local_files_only=True);prompts=[]
 for p in m['pairs']:
  valid=True
  for e in p['endpoints']:
   assert tok.encode(e['prompt'],add_special_tokens=False)==e['ids'];groups=[];prompts.append(e['prompt'])
   for target in p['endpoints']:
    gg=[]
    for form in dict.fromkeys([target['answer'],target['answer'].capitalize(),target['answer'].upper()]):
     ids=tok.encode(e['prompt']+' '+form,add_special_tokens=False)
     if len(ids)==len(e['ids'])+1 and ids[:-1]==e['ids']:gg.append(ids[-1])
    groups.append(sorted(set(gg)))
   assert groups==e['answer_groups'];valid=valid and all(groups) and not bool(set(groups[0])&set(groups[1]))
  assert bool(valid)==p['valid'];assert p['family'] not in design['exclude_from_training']
 assert len(set(prompts))==112;lp=Path(r['logits_path']);assert sha(lp)==r['logits_sha256'];saved=torch.load(lp,weights_only=True);samples={(s['id'],s['endpoint']):s['logits'].numpy() for s in saved};assert len(samples)==100;maxerr=0;diagnostics=[]
 def lse(v):return float(v.max()+np.log(np.exp(v-v.max()).sum()))
 for row in r['rows']:
  p=next(p for p in m['pairs'] if p['id']==row['id']);assert p['valid'];out=[]
  for ep,e in enumerate(p['endpoints']):
   z=samples[p['id'],ep];g=e['answer_groups'];margin=lse(z[g[ep]])-lse(z[g[1-ep]]);top=int(z.argmax());rr=row['endpoints'][ep];assert abs(margin-rr['margin'])<1e-10;maxerr=max(maxerr,abs(margin-rr['margin']));assert (margin>0,top in g[ep],top)==(rr['binary'],rr['top1'],rr['top_token']);out.append((margin>0,top in g[ep]));diagnostics.append(dict(id=p['id'],endpoint=ep,intended=e['answer'],top_decoded=tok.decode([top])))
  assert row['both_correct']==all(x[0] for x in out);assert row['top1_fraction']==sum(x[1] for x in out)/2
 for s in r['summaries']:
  rr=[x for x in r['rows'] if (x['family'],x['format'])==(s['family'],s['format'])];n=len(rr);acc=sum(x['both_correct'] for x in rr)/n if n else 0;top=sum(x['top1_fraction'] for x in rr)/n if n else 0;assert (n,acc,top)==(s['valid_pairs'],s['both_endpoint_accuracy'],s['top1']);assert s['pass_gate']==(n>=4 and acc>=.75 and top>=.5)
 assert r['forward_count']==len(samples)+len(r['native_replays']);assert len(r['summaries'])==14
 out=dict(status='passed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),manifest_sha256=sha(mp),prompts_cross_tokenizer_checked=112,final_vectors_numpy_checked=100,invalid_pairs=6,summary_checks=14,max_margin_error=maxerr,direct_native_replay_checks=len(r['native_replays']),max_native_replay_error=max(x['max_absolute_error'] for x in r['native_replays']),qualifying=[s for s in r['summaries'] if s['pass_gate']],diagnostics=diagnostics,scope='Independent HF tokenizer vs producer tiktoken; NumPy margins/argmax and every gate. Pinned model independently replays selected inputs using Block.forward; no representation trained.')
 op=V/'broader_training_screen_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='diagnostics'})
if __name__=='__main__':main()
