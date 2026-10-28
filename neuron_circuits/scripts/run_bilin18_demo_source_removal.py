"""H677: query-input versus demonstration-answer write removal."""
import json,time,hashlib,re
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_demo_source_removal'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_demonstration_competence_manifest.json','native':ART/'bilin18_demonstration_competence_captures.pt','screen':ART/'bilin18_demonstration_competence.json','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['native'],weights_only=False);screen=json.loads(paths['screen'].read_text());qualified={(c['family'],c['form']) for c in screen['cells'] if c['qualified']};tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);roles=[]
 for r in rows:
  p=r['prompt'];enc=tok(p,add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==r['ids']
  if r['form']=='arrow':
   matches=list(re.finditer(r'^(.+?) ->(.*)$',p,re.M));assert len(matches)==5;query=matches[-1].span(1);answers=[m.span(2) for m in matches[:-1]]
  else:
   inputs=list(re.finditer(r'^Input: (.+)$',p,re.M));ans=list(re.finditer(r'^Answer:(.*)$',p,re.M));assert len(inputs)==5 and len(ans)==5;query=inputs[-1].span(1);answers=[m.span(1) for m in ans[:-1]]
  assert p[query[0]:query[1]]==r['input'] and all(p[a:b].strip() for a,b in answers)
  def select(spans):return [k for k,(a,b) in enumerate(enc['offset_mapping']) if any(max(a,c)<min(b,d) and p[max(a,c):min(b,d)].strip() for c,d in spans)]
  qp=select([query]);dp=select(answers);assert qp and len(dp)==4 and not set(qp)&set(dp);roles.append(dict(query_positions=qp,demo_answer_positions=dp,query_span=query,demo_answer_spans=answers))
 (ART/f'{STEM}_roles.json').write_text(json.dumps(roles,indent=2));cfg=dict(experiment='H677',hashes=hashes,design='192 H676 prompts. Native baseline then complete removal of query-input head4, query-input all9heads, or all4demonstration-answer head4 writes at final attention position layer14. Role spans fixed by tokenizer offsets. Primary cells fixed by H676>=75% task/copy rule; all other cells retained.',limitations='Removes final-position attention writes from selected source tokens, not source representations or entire heads. Demonstration-answer and query interventions differ in token counts/norms; compare as source-specific necessity probes, not equal-strength importance scores. Full-vocabulary next-token competence only.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 att=model.blocks[14].attn;W=att.c_proj.weight.double();buckets=defaultdict(list)
 for i,r in enumerate(rows):buckets[len(r['ids'])].append(i)
 modes=['baseline','query_head4','query_all','demo_head4'];vectors={m:torch.empty(len(rows),1152,dtype=torch.float64) for m in modes[1:]};gates=[];replays=[];outputs={};forwards=0
 for mode in modes:
  zall=torch.empty(len(rows),model.lm_head.weight.shape[0])
  for inds in buckets.values():
   for st in range(0,len(inds),8):
    bb=inds[st:st+8]
    if mode=='baseline':
     def inspect(m,args,out):
      y,bus=args;B,T,_=y.shape;q,k,q2,k2=[mod(y).view(B,T,9,128) for mod in [att.c_q,att.c_k,att.c_q2,att.c_k2]];cos,sin=att.rotary(q);q,k,q2,k2=[model.tt.apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin) for v in [q,k,q2,k2]];coef=((q[:,-1,None]*k).sum(-1)/128)*((q2[:,-1,None]*k2).sum(-1)/128);values=(1-att.lamb)*att.c_v(y).view(B,T,9,128)+att.lamb*bus.view(B,T,9,128);weighted=coef[...,None].double()*values.double();pred=weighted.sum(1).reshape(B,1152)@W.T;truth=out[0][:,-1].double();replays.extend(((pred-truth).norm(dim=-1)/truth.norm(dim=-1)).cpu().tolist())
      for n,i in enumerate(bb):
       qp=roles[i]['query_positions'];dp=roles[i]['demo_answer_positions'];vectors['query_head4'][i]=(weighted[n,qp,4].sum(0)@W[:,512:640].T).cpu();vectors['query_all'][i]=(weighted[n,qp].sum(0).reshape(1152)@W.T).cpu();vectors['demo_head4'][i]=(weighted[n,dp,4].sum(0)@W[:,512:640].T).cpu();gates.append(dict(id=rows[i]['id'],query_head4_coefficients=coef[n,qp,4].cpu().tolist(),demo_head4_coefficients=coef[n,dp,4].cpu().tolist()))
     hook=att.register_forward_hook(inspect)
    else:
     patch=-vectors[mode][bb]
     def edit(m,args,out):
      y,b=out;y=y.clone();y[:,-1]+=patch.to(y);return y,b
     hook=att.register_forward_hook(edit)
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'));zall[bb]=z.cpu()
    finally:hook.remove()
    forwards+=len(bb)
  outputs[mode]=zall;print('COMPLETE',mode,flush=True)
 native=cap['native_logits'].double();lp0=native.log_softmax(-1);p0=lp0.exp();records=[]
 for mode,z in outputs.items():
  lp=z.double().log_softmax(-1)
  for i,r in enumerate(rows):
   target=r['answer_id'];records.append(dict(id=r['id'],family=r['family'],form=r['form'],control=r['control'],item=r['item'],qualified=(r['family'],r['form']) in qualified,mode=mode,target_ce=float(-lp[i,target]),target_ce_delta=float(lp0[i,target]-lp[i,target]),native_kl=float((p0[i]*(lp0[i]-lp[i])).sum()),top1_correct=int(z[i].argmax())==target,native_top1_correct=int(native[i].argmax())==target,removed_norm=0. if mode=='baseline' else float(vectors[mode][i].norm())))
 replay=float((outputs['baseline'].double()-native).abs().max());assert replay<1e-3 and max(replays)<1e-4
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},native_replay_max_abs=replay,attention_reconstruction_max_relative=max(replays),whole_model_forwards=forwards,finite=all(bool(torch.isfinite(v).all()) for v in outputs.values()),role_span_assertions_passed=True)
 torch.save(dict(logits=outputs,vectors=vectors),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,gates=gates,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits)
if __name__=='__main__':main()
