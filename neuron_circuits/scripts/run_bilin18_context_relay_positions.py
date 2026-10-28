"""H847 single-position localization of same-token layer4 contextual transfer."""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_context_relay_positions'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'data':ART/'bilin18_same_token_context_transfer_captures.pt','result':ART/'bilin18_same_token_context_transfer.json','examples':ART/'bilin18_candidate_task_screen_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H847','hashes':hashes,'design':'Posthoc localization at preMLP4 in all31 equal-length agreement/sentiment pairs, both directions. Every identical-token position at/after first differing token individually; no effect-based position filtering. Compare nonfinal/all same-position joint effects retained from H846.','scope':'Exploratory position selection on reused tasks, not confirmation of a discovered unit. Final token included as control. Recipient persistent streams and all other positions unchanged. Same-token context difference no lexical change at intervention site.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['data'],weights_only=False);caps,native=data['premlp'],data['native'];old=json.loads(paths['result'].read_text());ex=torch.load(paths['examples'],weights_only=False)['examples'];pairs=[r for r in old['records'] if r['layer']==4 and r['kind']=='raw' and r['mode']=='all_same' and r['family'] in ['agreement','sentiment']];jobs=[];tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 for r in pairs:
  i,j=r['i'],r['j'];aa,bb=ex[i]['ids'],ex[j]['ids'];first=next(p for p in range(len(aa)) if aa[p]!=bb[p])
  for p in range(first,len(aa)):
   if aa[p]==bb[p]:jobs.append({key:r[key] for key in ['id','family','split','recipient','donor','i','j','native_answer_gap']}|{'position':p,'from_end':p-len(aa),'token_id':aa[p],'token_text':tok.decode([aa[p]])})
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 def run(ids,repl,positions):
  x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
  for l,block in enumerate(model.blocks):
   raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att
   if l==4:pre=pre.clone();pre[torch.arange(len(ids),device='cuda'),positions]=repl.to(pre)
   x=pre+block.mlp(F.rms_norm(pre,(1152,)))
  return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30)
 records=[];ident=0.;actual=0
 for length in sorted({len(ex[r['i']]['ids']) for r in jobs}):
  jj=[r for r in jobs if len(ex[r['i']]['ids'])==length]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[r['i']]['ids'] for r in batch],device='cuda');positions=torch.tensor([r['position'] for r in batch],device='cuda');repl=torch.stack([caps[r['j']][4][r['position']] for r in batch]).cuda();y=run(ids,repl,positions);actual+=len(batch)
   if st==0:
    own=torch.stack([caps[r['i']][4][r['position']] for r in batch]).cuda();yy=run(ids,own,positions);actual+=len(batch);ident=max(ident,max(float((yy[k].double().cpu()-native[r['i']]).abs().max()) for k,r in enumerate(batch)))
   for k,r in enumerate(batch):
    i=r['i'];base=measure(native[i],ex[i]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;records.append(r|{'donor_directed_delta':sgn*(value-base)})
  print('DONE',length,flush=True)
 summary=[]
 for fam in ['agreement','sentiment']:
  for offset in sorted({r['from_end'] for r in records if r['family']==fam}):
   rr=[r for r in records if r['family']==fam and r['from_end']==offset];summary.append({'family':fam,'from_end':offset,'pairs':len(rr)//2,'tokens':sorted({r['token_text'] for r in rr}),'gap_fraction':sum(r['donor_directed_delta'] for r in rr)/sum(r['native_answer_gap'] for r in rr)})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':ident,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['donor_directed_delta']) for r in records)};assert audits['hashes_unchanged'] and ident<1e-4 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits}),flush=True)
if __name__=='__main__':main()
