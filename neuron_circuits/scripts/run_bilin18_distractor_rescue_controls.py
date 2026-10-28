#!/usr/bin/env python3
"""H547: distinguish removing distractor evidence from adding answer-name evidence."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,continuation_id
from run_bilin18_three_name_mapping import score
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_distractor_rescue_controls';torch.set_num_threads(6)
@torch.inference_mode()
def main():
 start=time.time();src=ART/'bilin18_three_person_binding.json';old=json.loads(src.read_text());cp=ART/'bilin18_three_person_binding_native_cache.pt';saved=torch.load(cp,weights_only=False,map_location='cpu');rows,cache=saved['rows'],saved['cache'];native={n['id']:n for n in old['native']};donors=['Zoe','Charles','Nancy','Edward','Helen','Harry','Betty','Louis']
 cfg={'experiment':'H547','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'head':[14,4],'site':'distractor','modes':['self','candidate_mean','unrelated_mean','keep_75','keep_50','keep_25','zero'],'unrelated_names':donors,'hypotheses':['Deleting or attenuating the distractor contribution improves early-distractor behavior without injecting any answer name.','An unrelated-name mean also improves behavior if removal of distractor identity, not answer-name addition, drives rescue.'],'scope':'Follow-up on reused H546 panel, selected after observing candidate-mean rescue. Conditions frozen before their execution. No fresh causal-superiority claim. Only persistent-bus contribution from distractor source to L14H4 final destination is altered; other paths and downstream computation live.','precision':'float32 compute after bfloat16-rounded loader'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');print('LOCKED',flush=True)
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;l,h=14,4
 def token_bus(t):
  x0=F.rms_norm(model.embedding(torch.tensor([t],device='cuda')),(model.config.n_embd,));x=model.blocks[0].lambdas.sum()*x0;return model.blocks[0].attn.c_v(F.rms_norm(x,(model.config.n_embd,))).reshape(H,D)
 donor_ids=[tok.encode(' '+s,add_special_tokens=False) for s in donors];assert all(len(x)==1 for x in donor_ids);assert not set(x[0] for x in donor_ids)&set(t for r in rows for t in r['candidate_ids']);unrelated=torch.stack([token_bus(x[0]) for x in donor_ids]).mean(0);means={};jobs=defaultdict(list)
 for r in rows:
  if r['item'] not in means:means[r['item']]=torch.stack([token_bus(t) for t in r['candidate_ids']]).mean(0)
  c=cache[r['id']];T=len(r['ids']);p=r['sites']['distractor'];attn=model.blocks[l].attn;q,k,q2,k2=[c[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_k','c_q2','c_k2')];cos,sin=attn.rotary(q);q,k=model.tt.apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin);q2,k2=model.tt.apply_rotary_emb(F.rms_norm(q2,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k2,(D,)),cos,sin);coef=(q[0,-1,h]@k[0,p,h]/D)*(q2[0,-1,h]@k2[0,p,h]/D)*attn.lamb;original=c['bus'][p,h].cuda();values={'self':original,'candidate_mean':means[r['item']][h],'unrelated_mean':unrelated[h],'keep_75':original*.75,'keep_50':original*.5,'keep_25':original*.25,'zero':torch.zeros_like(original)}
  for mode,value in values.items():delta=coef*F.linear(value-original,attn.c_proj.weight[:,h*D:(h+1)*D]);jobs[T].append((r,mode,delta.cpu()))
 result={'config':cfg,'interventions':[],'summary':{},'status':'running'};out=ART/f'{STEM}.json'
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 for T,jj in sorted(jobs.items()):
  for begin in range(0,len(jj),24):
   batch=jj[begin:begin+24];logits,_=run(model,torch.tensor([r['ids'] for r,m,d in batch],device='cuda'),patch=(l,torch.stack([d for r,m,d in batch])))
   for ix,(r,mode,d) in enumerate(batch):ss=score(logits[ix],r['candidate_ids'],r['answer'],r['query']);result['interventions'].append({'id':r['id'],'item':r['item'],'format':r['format'],'mode':mode,**ss,'correct_margin_delta':ss['correct_margin']-native[r['id']]['correct_margin'],'nonquery_margin_delta':ss['nonquery_margin']-native[r['id']]['nonquery_margin']})
  save()
 for fmt in old['competence']:
  result['summary'][fmt]={}
  for mode in cfg['modes']:
   cc=[c for c in result['interventions'] if c['format']==fmt and c['mode']==mode];result['summary'][fmt][mode]={'candidate_accuracy':float(np.mean([c['candidate_correct'] for c in cc])),'full_vocab_top1':float(np.mean([c['full_top1'] for c in cc])),'mean_correct_margin_delta':float(np.mean([c['correct_margin_delta'] for c in cc]))}
 replay={c['id']:c for c in old['interventions'] if c['layer']==14 and c['head']==4 and c['site']=='distractor' and c['mode']=='neutralize'}
 result['audits']={'self_max_margin_delta':max(abs(c['correct_margin_delta']) for c in result['interventions'] if c['mode']=='self'),'candidate_mean_replay_max_margin_error':max(abs(c['correct_margin']-replay[c['id']]['correct_margin']) for c in result['interventions'] if c['mode']=='candidate_mean'),'unrelated_donors_disjoint_from_candidates':True,'finite':all(np.isfinite(c['correct_margin']) for c in result['interventions']),'source_unchanged':cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',result['summary'],flush=True)
if __name__=='__main__':main()
