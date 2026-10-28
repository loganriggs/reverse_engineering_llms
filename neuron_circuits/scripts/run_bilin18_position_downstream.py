"""H551: locate downstream changes behind H550 copying prediction failure."""
import json,hashlib,time
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_position_downstream'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_position_task_transfer.json';old=json.loads(src.read_text());rows=[r for r in json.loads((ART/'bilin18_position_task_transfer_manifest.json').read_text()) if r['family']=='sequence_copy'];native={r['id']:r for r in old['native']};cfg=dict(experiment='H551',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),head=[8,3],selection='Posthoc diagnostic of H550 L8H3 copying frozen-prediction failure, all 16 prompts and four nonnative positions retained.',modes=['live','freeze_mlp','freeze_attention','freeze_both'],scope='Freeze whole attention writes at layers >8 and/or whole MLP writes at layers >=8 to native values. These broad clamps locate downstream dependence, not unique neuron mediation.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;l,h=8,3;attn=model.blocks[l].attn;result=dict(config=cfg,interventions=[],status='running');replay=[]
 for r in rows:
  ids=torch.tensor([r['ids']],device='cuda');writes={};handles=[]
  for layer,b in enumerate(model.blocks):
   if layer>l:
    def ah(m,args,out,layer=layer):writes[(layer,'attn')]=out[0].clone()
    handles.append(b.attn.register_forward_hook(ah))
   if layer>=l:
    def mh(m,args,out,layer=layer):writes[(layer,'mlp')]=out.clone()
    handles.append(b.mlp.register_forward_hook(mh))
  try:z,cc=run(model,ids,capture=True)
  finally:
   for handle in handles:handle.remove()
  base=measure(z[0],r['groups'])['margin'];replay.append(abs(base-native[r['id']]['margin']));T=len(r['ids']);p=r['source_position'];raw={n:cc[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_q2','c_k','c_k2')};cos,sin=attn.rotary(raw['c_q']);vec={}
  for n,v in raw.items():
   if n.startswith('c_k'):v=v[:,p:p+1].expand(-1,T,-1,-1)
   vec[n]=model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,:,h]
  def coef(pos):return float((vec['c_q'][-1]@vec['c_k'][pos]/D)*(vec['c_q2'][-1]@vec['c_k2'][pos]/D)*attn.lamb)
  core=F.linear(cc['bus'][0,p,h].cuda(),attn.c_proj.weight[:,h*D:(h+1)*D]);spec=[v for v in old['predictions'] if v['id']==r['id'] and v['layer']==l and v['mode']!='native'];batch=[]
  for v in spec:
   delta=(coef(v['assigned_position'])-coef(p))*core
   for mode in cfg['modes']:batch.append((v,mode,delta))
  # Each clamp mode gets one batch of its four positions.
  for mode in cfg['modes']:
   bb=[b for b in batch if b[1]==mode];handles=[]
   for layer,b in enumerate(model.blocks):
    if layer>l and mode in ('freeze_attention','freeze_both'):
     def ah(m,args,out,layer=layer):return writes[(layer,'attn')].expand(len(bb),-1,-1).clone(),out[1]
     handles.append(b.attn.register_forward_hook(ah))
    if layer>=l and mode in ('freeze_mlp','freeze_both'):
     def mh(m,args,out,layer=layer):return writes[(layer,'mlp')].expand(len(bb),-1,-1).clone()
     handles.append(b.mlp.register_forward_hook(mh))
   try:zz,_=run(model,ids.expand(len(bb),-1),patch=(l,torch.stack([d for v,m,d in bb])))
   finally:
    for handle in handles:handle.remove()
   for ix,(v,m,d) in enumerate(bb):delta=measure(zz[ix],r['groups'])['margin']-base;result['interventions'].append(dict(id=r['id'],position_mode=v['mode'],clamp=mode,margin_delta=delta,predicted_margin_delta=v['predicted_margin_delta']))
 result['summary']={}
 for mode in cfg['modes']:
  cc=[c for c in result['interventions'] if c['clamp']==mode];result['summary'][mode]=dict(sign_agreement=float(np.mean([np.sign(c['margin_delta'])==np.sign(c['predicted_margin_delta']) for c in cc])),mean_absolute_effect=float(np.mean([abs(c['margin_delta']) for c in cc])),prediction_mae=float(np.mean([abs(c['margin_delta']-c['predicted_margin_delta']) for c in cc])))
 oldlive={(c['id'],c['mode']):c['margin_delta'] for c in old['interventions'] if c['layer']==l};result['audits']=dict(native_replay_max_error=max(replay),live_replay_max_error=max(abs(c['margin_delta']-oldlive[c['id'],c['position_mode']]) for c in result['interventions'] if c['clamp']=='live'),frozen_prediction_max_error=max(abs(c['margin_delta']-c['predicted_margin_delta']) for c in result['interventions'] if c['clamp']=='freeze_both'),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=result['summary'],audits=result['audits']),indent=2),flush=True)
if __name__=='__main__':main()
