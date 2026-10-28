"""H565: prospective rotary-only prediction for physical gap insertion after source."""
import json,hashlib,time
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure,continuation_id
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_equal_length_gap_content'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_physical_position_prediction.json';tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);names=[('Aaron','Abby'),('Alan','Donna'),('Bruce','Diane'),('Carl','Ellen'),('Craig','Erin'),('Gary','Helen'),('Jerry','Joyce'),('Larry','Sharon')];gaps={'base':'','usual':' (as usual)','expected':' (as everyone expected)','fact':' (in fact)','once':' (for once)','said':' (as people said)','others':' (as others expected)'};rows=[]
 for i,(a,b) in enumerate(names):
  for fam in ('binding','copying'):
   for v,q in enumerate((a,b)):
    before=f'{a} and {b} went to the market. {q}' if fam=='binding' else f'He spoke to {q}. He spoke to {q}';after=' gave a parcel to' if fam=='binding' else '. He spoke to';baseid=f'{fam}-{i}-{v}';p=len(tok.encode(before,add_special_tokens=False))-1;answers=[b,a] if fam=='binding' else [a,b]
    for label,gap in gaps.items():
     text=before+gap+after;ids=tok.encode(text,add_special_tokens=False);gs=[[continuation_id(tok,text,' '+ans)] for ans in answers];assert all(g[0] is not None for g in gs);assert len(tok.encode(' '+q,add_special_tokens=False))==1 and ids[p]==tok.encode(' '+q,add_special_tokens=False)[0];rows.append(dict(id=baseid+'-'+label,baseid=baseid,family=fam,item=i,variant=v,gap=label,prompt=text,ids=ids,groups=gs,source_position=p))
 by={r['id']:r for r in rows}
 for r in rows:
  group=['usual','fact','once'] if r['gap'] in ('usual','fact','once') else ['expected','said','others'] if r['gap']!='base' else ['base']
  assert len({len(by[r['baseid']+'-'+g]['ids']) for g in group})==1
 cfg=dict(experiment='H566',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),heads=[[8,3],[14,4]],gaps=gaps,design='Reuse eight name pairs with three four-token and three five-token insertions plus base. Four new insertions chosen by token length before inference. Compare fixed-position content effects. Insert text strictly after selected name. Fixed heads, no case selection. Compute rotary-only predictions from base prompts before evaluating edited prompts.',modes=['self','phase_prediction','base_coefficient','zero_edge'],scope='Physical edits change query context as well as distance. No claim of pure position manipulation. Coefficient interventions touch only the selected persistent-bus edge; other prompt-induced changes remain.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(mp.read_bytes()).hexdigest();model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for param in mod.parameters():param.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;cache={};native={};predictions=[]
 def capture(rr):
  buckets=defaultdict(list)
  for r in rr:buckets[len(r['ids'])].append(r)
  for T,bb in buckets.items():
   for st in range(0,len(bb),16):
    batch=bb[st:st+16];z,c=run(model,torch.tensor([r['ids'] for r in batch],device='cuda'),capture=True)
    for ix,r in enumerate(batch):cache[r['id']]={n:v[ix].clone() for n,v in c.items()};native[r['id']]=measure(z[ix],r['groups'])
 def coef(c,l,h,qpos,kpos,T,sourcep):
  attn=model.blocks[l].attn;raw={n:c[(l,n)][-1 if n.startswith('c_q') else sourcep].cuda().view(1,1,H,D).expand(1,T,H,D) for n in ('c_q','c_q2','c_k','c_k2')};cos,sin=attn.rotary(raw['c_q']);vv={n:model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,qpos if n.startswith('c_q') else kpos,h] for n,v in raw.items()};return float((vv['c_q']@vv['c_k']/D)*(vv['c_q2']@vv['c_k2']/D)*attn.lamb)
 capture([r for r in rows if r['gap']=='base'])
 for r in rows:
  base=by[r['baseid']+'-base'];c=cache[base['id']];p=r['source_position'];assert r['ids'][:p+1]==base['ids'][:p+1]
  for l,h in cfg['heads']:predictions.append(dict(id=r['id'],family=r['family'],gap=r['gap'],layer=l,head=h,predicted_coefficient=coef(c,l,h,len(r['ids'])-1,p,len(r['ids']),p),base_coefficient=coef(c,l,h,len(base['ids'])-1,p,len(base['ids']),p)))
 pp=ART/f'{STEM}_predictions.json';pp.write_text(json.dumps(predictions,indent=2));ph=hashlib.sha256(pp.read_bytes()).hexdigest();print('PREDICTIONS LOCKED BEFORE EDITED EVALUATION',len(predictions),flush=True)
 capture([r for r in rows if r['gap']!='base']);result=dict(config=cfg,native=[dict(id=r['id'],family=r['family'],gap=r['gap'],variant=r['variant'],**native[r['id']]) for r in rows],geometry=[],interventions=[],status='running');jobs=defaultdict(list);keyerrors=[]
 for rec in predictions:
  r=by[rec['id']];base=by[r['baseid']+'-base'];p=r['source_position'];l,h=rec['layer'],rec['head'];c=cache[r['id']];bc=cache[base['id']];actual=coef(c,l,h,len(r['ids'])-1,p,len(r['ids']),p);context_only=coef(c,l,h,len(base['ids'])-1,p,len(r['ids']),p);result['geometry'].append(dict(rec,actual_coefficient=actual,context_only_coefficient=context_only,source_position=p,base_final_position=len(base['ids'])-1,edited_final_position=len(r['ids'])-1))
  for n in ('c_k','c_k2'):keyerrors.append(float((c[(l,n)][p]-bc[(l,n)][p]).abs().max()))
  core=F.linear(c['bus'][p,h].cuda(),model.blocks[l].attn.c_proj.weight[:,h*D:(h+1)*D])
  for mode,co in [('self',actual),('phase_prediction',rec['predicted_coefficient']),('base_coefficient',rec['base_coefficient']),('zero_edge',0.)]:jobs[len(r['ids']),l].append((r,h,mode,((co-actual)*core).cpu()))
 for (T,l),jj in jobs.items():
  for st in range(0,len(jj),24):
   bb=jj[st:st+24];z,_=run(model,torch.tensor([r['ids'] for r,h,m,d in bb],device='cuda'),patch=(l,torch.stack([d for r,h,m,d in bb])))
   for ix,(r,h,mode,d) in enumerate(bb):ss=measure(z[ix],r['groups']);result['interventions'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],layer=l,head=h,mode=mode,margin_delta=ss['margin']-native[r['id']]['margin'],candidate_correct=(ss['margin']>0)==(r['variant']==0),full_top1=ss['top1'][r['variant']]))
 result['summary']={}
 for fam in ('binding','copying'):
  result['summary'][fam]={}
  for gap in gaps:
   nn=[r for r in result['native'] if r['family']==fam and r['gap']==gap];entry=dict(candidate_accuracy=float(np.mean([(r['margin']>0)==(r['variant']==0) for r in nn])),full_top1=float(np.mean([r['top1'][r['variant']] for r in nn])),heads={})
   for l,h in cfg['heads']:
    gg=[g for g in result['geometry'] if g['family']==fam and g['gap']==gap and g['layer']==l];pred_err=np.mean([abs(g['predicted_coefficient']-g['actual_coefficient']) for g in gg]);base_err=np.mean([abs(g['base_coefficient']-g['actual_coefficient']) for g in gg]);nz=[g for g in gg if abs(g['actual_coefficient']-g['base_coefficient'])>1e-4];entry['heads'][f'L{l}H{h}']=dict(phase_mae=float(pred_err),unchanged_mae=float(base_err),phase_error_ratio=float(pred_err/base_err) if base_err>1e-8 else None,direction_agreement=float(np.mean([np.sign(g['predicted_coefficient']-g['base_coefficient'])==np.sign(g['actual_coefficient']-g['base_coefficient']) for g in nz])) if nz else None,modes={})
    for mode in cfg['modes']:
     cc=[c for c in result['interventions'] if c['family']==fam and c['gap']==gap and c['layer']==l and c['mode']==mode];entry['heads'][f'L{l}H{h}']['modes'][mode]=dict(mean_abs_margin_change=float(np.mean([abs(c['margin_delta']) for c in cc])),candidate_accuracy=float(np.mean([c['candidate_correct'] for c in cc])))
   result['summary'][fam][gap]=entry
 result['audits']=dict(source_key_max_error=max(keyerrors),self_max_margin_error=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='self'),predictions_unchanged=ph==hashlib.sha256(pp.read_bytes()).hexdigest(),equal_length_groups_verified=True,manifest_unchanged=mh==hashlib.sha256(mp.read_bytes()).hexdigest(),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']));result['status']='complete';result['elapsed_seconds']=time.time()-start;torch.save({'rows':rows,'cache':cache,'native':native},ART/f'{STEM}_native_cache.pt');(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['audits']),flush=True)
if __name__=='__main__':main()
