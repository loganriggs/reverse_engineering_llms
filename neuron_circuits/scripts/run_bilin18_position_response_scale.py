"""H555: diverse-task virtual position sweep and frozen-downstream predictions."""
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
STEM='bilin18_position_response_scale'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);rows=[];excluded=[]
 names=[('Arthur','Olivia'),('Albert','Sophia'),('Fred','Maria'),('Philip','Nicole'),('Richard','Amy'),('Steven','Karen'),('Tony','Julie'),('Vincent','Angela')]
 values=[('red','blue'),('green','yellow'),('black','white'),('orange','purple'),('gold','silver'),('pink','brown'),('gray','violet'),('cyan','tan')]
 for item,((a,b),(v,w)) in enumerate(zip(names,values)):
  for fam,prompts,answers in [
   ('color_prose',[f'The code was {x}. The code was {x}. The code was' for x in (v,w)],[v,w]),
   ('color_record',[f'record {x} | record {x} | record' for x in (v,w)],[v,w]),
   ('name_prose',[f'The code was {x}. The code was {x}. The code was' for x in (a,b)],[a,b]),
   ('name_record',[f'record {x} | record {x} | record' for x in (a,b)],[a,b]),
   ('binding',[f'{a} and {b} entered the room. {g} passed the note to' for g in (a,b)],[b,a])]:
   ids=[tok.encode(t,add_special_tokens=False) for t in prompts]
   if len(ids[0])!=len(ids[1]):excluded.append(f'{fam}-{item}');continue
   dif=[i for i,(x,y) in enumerate(zip(*ids)) if x!=y];p=dif[-1]
   for v in (0,1):
    gs=[[continuation_id(tok,prompts[v],' '+ans)] for ans in answers];assert all(g[0] is not None for g in gs);rows.append(dict(id=f'{fam}-{item}-{v}',pair=f'{fam}-{item}',family=fam,variant=v,ids=ids[v],prompt=prompts[v],groups=gs,source_position=p))
 src=ART/'bilin18_position_vocabulary_control.json';cfg=dict(experiment='H555',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),heads=[[8,3]],selected_mlp=8,clamps=['live','single_8','freeze_mlp','freeze_both'],scales=[-1.,-.25,-.0625,0.,.0625,.25,1.],design='Amplitude follow-up on the unchanged H554 panel. Use only the virtual destination-position edge delta, scale it positively/negatively, and zero-control it. Cross the same eight name pairs used in binding into the identical prose-code and record-list copying templates used for colors. Reuse color/binding anchors; no outcome filtering. Layer8 selected by H552 minimum single-layer prediction MAE before new evaluation. Five virtual key positions: start, halfway toward start, native, halfway toward destination, destination. Only one bus edge changes.',predictions=['If the reversal is a locally stable downstream response, it persists at 1/16 amplitude and both signs.','If it needs a large nonlinear excursion, sign disagreement or response per unit amplitude changes sharply near zero.'],scope='Source is last differing token. Virtual phase changes do not physically move words. Tasks not necessarily mediated by these heads. Physical position validation remains outstanding.',precision='float32 after bf16-rounded loader')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(mp.read_bytes()).hexdigest();print('LOCKED',len(rows),'excluded',excluded,flush=True)
 model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 cache={};native={};buckets=defaultdict(list)
 for r in rows:buckets[len(r['ids'])].append(r)
 for T,rr in buckets.items():
  for begin in range(0,len(rr),16):
   bb=rr[begin:begin+16];writes={};handles=[]
   for layer,block in enumerate(model.blocks):
    if layer>=8:
     def mlp_hook(m,args,out,layer=layer):writes[('write_mlp',layer)]=out.cpu()
     handles.append(block.mlp.register_forward_hook(mlp_hook))
    if layer>8:
     def ah(m,args,out,layer=layer):writes[('write_attn',layer)]=out[0].cpu()
     handles.append(block.attn.register_forward_hook(ah))
   try:z,c=run(model,torch.tensor([r['ids'] for r in bb],device='cuda'),capture=True)
   finally:
    for handle in handles:handle.remove()
   c.update(writes)
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
   for scale in cfg['scales']:
    mode='native' if scale==0 else str(scale);pos=T-1;co=coeff(pos);delta=scale*(co-nc)*core;last=c['last'].cuda()+gain*delta;z=30*torch.tanh(model.lm_head(F.rms_norm(last,(model.config.n_embd,))).float()/30);pred=measure(z,r['groups'])['margin']-native[r['id']]['margin'];rec=dict(id=r['id'],family=r['family'],layer=l,head=h,mode=mode,scale=scale,source_position=p,assigned_position=pos,native_coefficient=nc,coefficient=co,predicted_margin_delta=pred);result['predictions'].append(rec);
    for clamp in cfg['clamps']:jobs[(T,l,clamp)].append((r,dict(rec,clamp=clamp),delta.cpu()))
 torch.save({'rows':rows,'cache':cache,'native':native},ART/f'{STEM}_native_cache.pt')
 out=ART/f'{STEM}.json';out.write_text(json.dumps(result,indent=2));prediction_hash=hashlib.sha256(json.dumps(result['predictions'],sort_keys=True).encode()).hexdigest();print('PREDICTIONS LOCKED',len(result['predictions']),flush=True)
 for (T,l,clamp),jj in jobs.items():
  for begin in range(0,len(jj),24):
   bb=jj[begin:begin+24];handles=[]
   for layer,block in enumerate(model.blocks):
    if layer>=8 and (clamp in ('freeze_mlp','freeze_both') or (clamp=='single_8' and layer==8)):
     def mlp_hook(m,args,out,layer=layer):return torch.stack([cache[r['id']][('write_mlp',layer)] for r,rec,d in bb]).to(out)
     handles.append(block.mlp.register_forward_hook(mlp_hook))
    if layer>8 and clamp=='freeze_both':
     def ah(m,args,out,layer=layer):return torch.stack([cache[r['id']][('write_attn',layer)] for r,rec,d in bb]).to(out[0]),out[1]
     handles.append(block.attn.register_forward_hook(ah))
   try:z,_=run(model,torch.tensor([r['ids'] for r,rec,d in bb],device='cuda'),patch=(l,torch.stack([d for r,rec,d in bb])))
   finally:
    for handle in handles:handle.remove()
   for ix,(r,rec,d) in enumerate(bb):ss=measure(z[ix],r['groups']);result['interventions'].append(dict(**rec,margin_delta=ss['margin']-native[r['id']]['margin'],candidate_correct=(ss['margin']>0)==(r['variant']==0),full_top1=ss['top1'][r['variant']]))
 result['summary']={}
 for fam in sorted(set(r['family'] for r in rows)):
  rr=[r for r in rows if r['family']==fam];nn=[native[r['id']] for r in rr];result['summary'][fam]=dict(n=len(rr),native_candidate_accuracy=float(np.mean([(n['margin']>0)==(r['variant']==0) for r,n in zip(rr,nn)])),native_full_top1=float(np.mean([n['top1'][r['variant']] for r,n in zip(rr,nn)])),heads={})
  for clamp in cfg['clamps']:
   result['summary'][fam]['heads'][clamp]={}
   for scale in cfg['scales']:
    if scale==0:continue
    cc=[c for c in result['interventions'] if c['family']==fam and c['clamp']==clamp and c['scale']==scale];nonzero=[c for c in cc if abs(c['predicted_margin_delta'])>1e-4 and abs(c['margin_delta'])>1e-4];result['summary'][fam]['heads'][clamp][str(scale)]=dict(sign_agreement=float(np.mean([np.sign(c['predicted_margin_delta'])==np.sign(c['margin_delta']) for c in nonzero])) if nonzero else None,n_sign=len(nonzero),mean_absolute_effect_per_scale=float(np.mean([abs(c['margin_delta']/scale) for c in cc])),prediction_mae_per_scale=float(np.mean([abs((c['predicted_margin_delta']-c['margin_delta'])/scale) for c in cc])))
 result['audits']=dict(self_max_margin_delta=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='native'),predictions_unchanged=prediction_hash==hashlib.sha256(json.dumps(result['predictions'],sort_keys=True).encode()).hexdigest(),manifest_unchanged=mh==hashlib.sha256(mp.read_bytes()).hexdigest(),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']));result['audits']['frozen_prediction_max_error']=max(abs(c['margin_delta']-c['predicted_margin_delta']) for c in result['interventions'] if c['clamp']=='freeze_both');result['status']='complete';result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=result['summary'],audits=result['audits']),indent=2),flush=True)
if __name__=='__main__':main()
