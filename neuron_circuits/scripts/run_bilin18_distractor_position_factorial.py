"""H549: Q content x K content x assigned rotary position, local bus-edge intervention."""
import json,hashlib,itertools,time
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_three_name_mapping import score
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_distractor_position_factorial'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_three_person_binding.json';cp=ART/'bilin18_three_person_binding_native_cache.pt';old=json.loads(src.read_text());saved=torch.load(cp,weights_only=False);cache=saved['cache'];rows=[r for r in saved['rows'] if r['format'] in ('distractor_first','distractor_middle')];pairs={}
 for r in rows:pairs.setdefault((r['item'],r['rotation'],r['query']),{})[r['format'].split('_')[-1]]=r
 cfg=dict(experiment='H549',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),cache_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),head=[14,4],design='48 matched first/middle pairs; full Q-content x K-content x assigned key rotary position factorial. All 8 combinations applied to both native recipients, plus deletion. No fitting or filtering.',scope='Changes only the distractor persistent-bus contribution at the final destination. Virtual rotary reassignment is not a physical token move or a complete key swap. Reused diagnostic panel.',precision='float32 computation with bf16-rounded weights')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));print('LOCKED',len(pairs),flush=True)
 model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;l,h=14,4;attn=model.blocks[l].attn;oldnat={v['id']:v for v in old['native']};oldgeo={g['id']:g['coefficient'] for g in old['geometry'] if g['layer']==l and g['head']==h and g['site']=='distractor'};result=dict(config=cfg,geometry=[],interventions=[],status='running');jobs=[];errors=[];buserrors=[]
 for pair,rr in pairs.items():
  a,b=rr['first'],rr['middle'];T=len(a['ids']);assert T==len(b['ids']) and a['ids'][-1]==b['ids'][-1];pos={s:r['sites']['distractor'] for s,r in rr.items()};assert a['ids'][pos['first']]==b['ids'][pos['middle']];vec={}
  for s,r in rr.items():
   c=cache[r['id']];raw={n:c[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_q2','c_k','c_k2')};cos,sin=attn.rotary(raw['c_q']);vec[s]={}
   for n,v in raw.items():
    if n.startswith('c_k'):v=v[:,pos[s]:pos[s]+1].expand(-1,T,-1,-1)
    vec[s][n]=model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,:,h]
  co={}
  for q,k,p in itertools.product(('first','middle'),repeat=3):
   f1=float(vec[q]['c_q'][-1]@vec[k]['c_k'][pos[p]]/D);f2=float(vec[q]['c_q2'][-1]@vec[k]['c_k2'][pos[p]]/D);mode=f'{q}/{k}/{p}';co[mode]=f1*f2*float(attn.lamb);result['geometry'].append(dict(pair=list(pair),mode=mode,factor1=f1,factor2=f2,coefficient=co[mode]))
  buserrors.append(float((cache[a['id']]['bus'][pos['first'],h]-cache[b['id']]['bus'][pos['middle'],h]).abs().max()))
  for s,r in rr.items():
   nativecoef=co['/'.join([s]*3)];errors.append(abs(nativecoef-oldgeo[r['id']]));core=F.linear(cache[r['id']]['bus'][pos[s],h].cuda(),attn.c_proj.weight[:,h*D:(h+1)*D])
   for mode,coef in list(co.items())+[('zero',0.)]:jobs.append((r,s,mode,((coef-nativecoef)*core).cpu()))
 for begin in range(0,len(jobs),24):
  batch=jobs[begin:begin+24];z,_=run(model,torch.tensor([r['ids'] for r,s,m,d in batch],device='cuda'),patch=(l,torch.stack([d for r,s,m,d in batch])))
  for ix,(r,s,mode,d) in enumerate(batch):
   ss=score(z[ix],r['candidate_ids'],r['answer'],r['query']);result['interventions'].append(dict(id=r['id'],recipient=s,mode=mode,**ss,margin_delta=ss['correct_margin']-oldnat[r['id']]['correct_margin']))
  if begin%192==0:print('DONE',begin+len(batch),flush=True)
 result['summary']={}
 for s in ('first','middle'):
  result['summary'][s]={}
  for mode in list(co)+['zero']:
   cc=[c for c in result['interventions'] if c['recipient']==s and c['mode']==mode];gg=[g['coefficient'] for g in result['geometry'] if g['mode']==mode];result['summary'][s][mode]=dict(candidate_accuracy=float(np.mean([c['candidate_correct'] for c in cc])),full_vocab_top1=float(np.mean([c['full_top1'] for c in cc])),mean_margin_delta=float(np.mean([c['margin_delta'] for c in cc])),mean_coefficient=float(np.mean(gg)) if gg else 0.)
 result['audits']=dict(native_coefficient_max_error=max(errors),bus_max_error=max(buserrors),self_max_margin_delta=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='/'.join([c['recipient']]*3)),finite=all(np.isfinite(c['correct_margin']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),cache_unchanged=cfg['cache_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=result['summary'],audits=result['audits']),indent=2),flush=True)
if __name__=='__main__':main()
