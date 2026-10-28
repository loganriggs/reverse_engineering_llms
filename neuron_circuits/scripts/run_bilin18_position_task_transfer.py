"""H550: diverse-task virtual position sweep and frozen-downstream predictions."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,suite,groups,measure,continuation_id
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_position_task_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);rows=[];excluded=[]
 for r in suite():
  if r['split']!='transfer' or r['family'] not in ('entity_binding','sequence_copy','agreement','sentiment'):continue
  ids=[tok.encode(t,add_special_tokens=False) for t in r['prompts']]
  if len(ids[0])!=len(ids[1]):excluded.append(r['id']);continue
  dif=[i for i,(a,b) in enumerate(zip(*ids)) if a!=b];p=dif[-1]
  for v in (0,1):
   gs=[[continuation_id(tok,r['prompts'][v],s) for s in g] for g in groups(r['family'],r['answers'])];gs=[[t for t in g if t is not None] for g in gs];assert all(gs);rows.append(dict(id=r['id']+f'-{v}',pair=r['id'],family=r['family'],variant=v,ids=ids[v],groups=gs,source_position=p))
 src=ART/'bilin18_distractor_position_factorial.json';cfg=dict(experiment='H550',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),heads=[[14,4],[8,3]],design='All token-valid H533 transfer pairs in binding, copying, agreement, sentiment; reuse prompts without outcome selection. Five virtual key positions: start, halfway toward start, native, halfway toward destination, destination. Only one bus edge changes.',predictions=['Moving key position nearer the destination weakens absolute coefficient, as in H549; test per family/head without fitting.','A frozen-downstream final-residual readout predicts live intervention margin changes; test signed and magnitude agreement.'],scope='Source is last differing token. Virtual phase changes do not physically move words. Tasks not necessarily mediated by these heads. Physical position validation remains outstanding.',precision='float32 after bf16-rounded loader')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(mp.read_bytes()).hexdigest();print('LOCKED',len(rows),'excluded',excluded,flush=True)
 model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 cache={};native={};buckets=defaultdict(list)
 for r in rows:buckets[len(r['ids'])].append(r)
 for T,rr in buckets.items():
  for begin in range(0,len(rr),16):
   bb=rr[begin:begin+16];z,c=run(model,torch.tensor([r['ids'] for r in bb],device='cuda'),capture=True)
   for ix,r in enumerate(bb):cache[r['id']]={n:v[ix] for n,v in c.items()};native[r['id']]=measure(z[ix],r['groups'])
 result=dict(config=cfg,manifest_sha256=mh,excluded_pairs=excluded,native=[dict(id=r['id'],family=r['family'],variant=r['variant'],**native[r['id']]) for r in rows],predictions=[],interventions=[],status='predictions_locked');jobs=defaultdict(list);H=model.config.n_head;D=model.config.n_embd//H
 for r in rows:
  c=cache[r['id']];T=len(r['ids']);p=r['source_position'];positions=dict(start=0,earlier=p//2,native=p,nearer=(p+T-1)//2,destination=T-1)
  for l,h in cfg['heads']:
   attn=model.blocks[l].attn;raw={n:c[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_q2','c_k','c_k2')};cos,sin=attn.rotary(raw['c_q']);vec={}
   for n,v in raw.items():
    if n.startswith('c_k'):v=v[:,p:p+1].expand(-1,T,-1,-1)
    vec[n]=model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,:,h]
   def coeff(pos):return float((vec['c_q'][-1]@vec['c_k'][pos]/D)*(vec['c_q2'][-1]@vec['c_k2'][pos]/D)*attn.lamb)
   nc=coeff(p);core=F.linear(c['bus'][p,h].cuda(),attn.c_proj.weight[:,h*D:(h+1)*D]);gain=1.
   for b in model.blocks[l+1:]:gain*=float(b.lambdas[0])
   for mode,pos in positions.items():
    co=coeff(pos);delta=(co-nc)*core;last=c['last'].cuda()+gain*delta;z=30*torch.tanh(model.lm_head(F.rms_norm(last,(model.config.n_embd,))).float()/30);pred=measure(z,r['groups'])['margin']-native[r['id']]['margin'];rec=dict(id=r['id'],family=r['family'],layer=l,head=h,mode=mode,source_position=p,assigned_position=pos,native_coefficient=nc,coefficient=co,predicted_margin_delta=pred);result['predictions'].append(rec);jobs[(T,l)].append((r,rec,delta.cpu()))
 out=ART/f'{STEM}.json';out.write_text(json.dumps(result,indent=2));prediction_hash=hashlib.sha256(json.dumps(result['predictions'],sort_keys=True).encode()).hexdigest();print('PREDICTIONS LOCKED',len(result['predictions']),flush=True)
 for (T,l),jj in jobs.items():
  for begin in range(0,len(jj),24):
   bb=jj[begin:begin+24];z,_=run(model,torch.tensor([r['ids'] for r,rec,d in bb],device='cuda'),patch=(l,torch.stack([d for r,rec,d in bb])))
   for ix,(r,rec,d) in enumerate(bb):ss=measure(z[ix],r['groups']);result['interventions'].append(dict(**rec,margin_delta=ss['margin']-native[r['id']]['margin'],candidate_correct=(ss['margin']>0)==(r['variant']==0),full_top1=ss['top1'][r['variant']]))
 result['summary']={}
 for fam in sorted(set(r['family'] for r in rows)):
  rr=[r for r in rows if r['family']==fam];nn=[native[r['id']] for r in rr];result['summary'][fam]=dict(n=len(rr),native_candidate_accuracy=float(np.mean([(n['margin']>0)==(r['variant']==0) for r,n in zip(rr,nn)])),native_full_top1=float(np.mean([n['top1'][r['variant']] for r,n in zip(rr,nn)])),heads={})
  for l,h in cfg['heads']:
   cc=[c for c in result['interventions'] if c['family']==fam and c['layer']==l and c['mode']!='native'];near=[c for c in cc if c['mode']=='nearer'];nonzero=[c for c in cc if abs(c['predicted_margin_delta'])>1e-4 and abs(c['margin_delta'])>1e-4];result['summary'][fam]['heads'][f'L{l}H{h}']=dict(nearer_weakens_fraction=float(np.mean([abs(c['coefficient'])<abs(c['native_coefficient']) for c in near])),sign_agreement=float(np.mean([np.sign(c['predicted_margin_delta'])==np.sign(c['margin_delta']) for c in nonzero])) if nonzero else None,n_sign=len(nonzero),mean_absolute_live_effect=float(np.mean([abs(c['margin_delta']) for c in cc])),prediction_mae=float(np.mean([abs(c['predicted_margin_delta']-c['margin_delta']) for c in cc])))
 result['audits']=dict(self_max_margin_delta=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='native'),predictions_unchanged=prediction_hash==hashlib.sha256(json.dumps(result['predictions'],sort_keys=True).encode()).hexdigest(),manifest_unchanged=mh==hashlib.sha256(mp.read_bytes()).hexdigest(),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']));result['status']='complete';result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=result['summary'],audits=result['audits']),indent=2),flush=True)
if __name__=='__main__':main()
