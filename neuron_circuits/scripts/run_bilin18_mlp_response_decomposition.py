"""H556: exact normalized-input bilinear response and causal component replacements."""
import json,time,hashlib
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_mlp_response_decomposition'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_position_response_scale.json';old=json.loads(src.read_text());saved=torch.load(ART/'bilin18_position_vocabulary_control_native_cache.pt',weights_only=False);rows=saved['rows'];cache=saved['cache'];native=saved['native'];cfg=dict(experiment='H556',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),head=[8,3],layer=8,scales=[-.0625,.0625,-1.,1.],modes=['full','linear_normalized','quadratic_only','fixed_norm','tangent'],design='Reuse all H554 prompts. Scale destination-phase edge delta; replace only the final-position L8 MLP output response with algebraic components. Later computation live.',hypotheses=['Linear cross terms in normalized input dominate local bilinear response.','Changing the RMS denominator contributes materially to the response.','Exact normalized-input decomposition must reconstruct full MLP response before causal interpretation.'],scope='Selected same-layer MLP only; no claim it explains all downstream transformation.',precision='float32 computation, bf16-rounded weights; rotary caches bf16')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));print('LOCKED',len(rows),flush=True);model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 assert not model.config.gated
 mlp=model.blocks[8].mlp;block=model.blocks[8];D=model.config.n_embd//model.config.n_head;h=3;result=dict(config=cfg,geometry=[],interventions=[],status='running');jobs=defaultdict(list);stored={};errors=[];inputerrors=[]
 pred={v['id']:v for v in json.loads((ART/'bilin18_position_vocabulary_control.json').read_text())['predictions'] if v['mode']=='destination'}
 for r in rows:
  cc={}
  def pre(m,args):cc['x']=args[0][:,-1].clone();cc['x0']=args[2][:,-1].clone()
  def ah(m,args,out):cc['y']=out[0][:,-1].clone()
  def mp(m,args):cc['z']=args[0][:,-1].clone()
  hh=[block.register_forward_pre_hook(pre),block.attn.register_forward_hook(ah),mlp.register_forward_pre_hook(mp)]
  try:zz,_=run(model,torch.tensor([r['ids']],device='cuda'))
  finally:
   for handle in hh:handle.remove()
  x=(block.lambdas[0]*cc['x']+block.lambdas[1]*cc['x0']+cc['y'])[0];z0=F.rms_norm(x,(x.numel(),));inputerrors.append(float((z0-cc['z'][0]).abs().max()));y0=mlp(z0);p=r['source_position'];v=pred[r['id']];delta0=(v['coefficient']-v['native_coefficient'])*F.linear(cache[r['id']]['bus'][p,h].cuda(),block.attn.c_proj.weight[:,h*D:(h+1)*D]);stored[r['id']]=dict(x=x.cpu(),delta=delta0.cpu());eps=torch.finfo(x.dtype).eps;s0=(x.square().mean()+eps).sqrt()
  def lin(e):return mlp.Down(mlp.Left(z0)*mlp.Right(e)+mlp.Left(e)*mlp.Right(z0))
  for scale in cfg['scales']:
   delta=scale*delta0;z1=F.rms_norm(x+delta,(x.numel(),));e=z1-z0;linear=lin(e);quad=mlp.Down(mlp.Left(e)*mlp.Right(e));full=mlp(z1)-y0;fixed=mlp((x+delta)/s0)-y0;tangent=lin(delta/s0-x*(x*delta).mean()/s0**3);err=float((linear+quad-full).abs().max());errors.append(err);den=full.norm().clamp_min(1e-12);result['geometry'].append(dict(id=r['id'],family=r['family'],scale=scale,response_norm=float(den),quadratic_relative_norm=float(quad.norm()/den),fixed_norm_relative_error=float((fixed-full).norm()/den),tangent_relative_error=float((tangent-full).norm()/den),decomposition_relative_error=float((linear+quad-full).norm()/den),decomposition_max_error=err,delta_relative_input_norm=float(delta.norm()/x.norm())))
   for mode,response in [('full',full),('linear_normalized',linear),('quadratic_only',quad),('fixed_norm',fixed),('tangent',tangent)]:jobs[len(r['ids'])].append((r,scale,mode,delta.cpu(),(y0+response).cpu()))
 torch.save(stored,ART/f'{STEM}_inputs.pt');print('COMPONENTS READY',sum(map(len,jobs.values())),flush=True)
 for T,jj in jobs.items():
  for begin in range(0,len(jj),24):
   bb=jj[begin:begin+24]
   def hook(m,args,out):out=out.clone();out[:,-1]=torch.stack([y for r,s,m,d,y in bb]).to(out);return out
   handle=mlp.register_forward_hook(hook)
   try:zz,_=run(model,torch.tensor([r['ids'] for r,s,m,d,y in bb],device='cuda'),patch=(8,torch.stack([d for r,s,m,d,y in bb])))
   finally:handle.remove()
   for ix,(r,scale,mode,d,y) in enumerate(bb):ss=measure(zz[ix],r['groups']);result['interventions'].append(dict(id=r['id'],family=r['family'],scale=scale,mode=mode,margin_delta=ss['margin']-native[r['id']]['margin']))
 full={(c['id'],c['scale']):c['margin_delta'] for c in result['interventions'] if c['mode']=='full'};oldlive={(c['id'],c['scale']):c['margin_delta'] for c in old['interventions'] if c['clamp']=='live'};result['summary']={}
 for fam in sorted({r['family'] for r in rows}):
  result['summary'][fam]={}
  for scale in cfg['scales']:
   gg=[g for g in result['geometry'] if g['family']==fam and g['scale']==scale];entry={k:float(np.mean([g[k] for g in gg])) for k in ('quadratic_relative_norm','fixed_norm_relative_error','tangent_relative_error','delta_relative_input_norm')};entry['modes']={}
   for mode in cfg['modes']:
    cc=[c for c in result['interventions'] if c['family']==fam and c['scale']==scale and c['mode']==mode];entry['modes'][mode]=dict(mae_to_full=float(np.mean([abs(c['margin_delta']-full[c['id'],scale]) for c in cc])),mean_absolute_effect=float(np.mean([abs(c['margin_delta']) for c in cc])))
   result['summary'][fam][str(scale)]=entry
 result['audits']=dict(input_normalization_max_error=max(inputerrors),decomposition_max_error=max(errors),full_live_replay_max_error=max(abs(v-oldlive[k]) for k,v in full.items()),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['audits']),flush=True)
if __name__=='__main__':main()
