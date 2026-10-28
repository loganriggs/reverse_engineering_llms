"""H573: reconstruct L14 raw query edits and test single upstream write clamps."""
import json,hashlib,time
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_query_producer_group_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_query_write_construction.json';discovery=json.loads(src.read_text());old=json.loads((ART/'bilin18_query_punctuation_copy_retest.json').read_text());saved=torch.load(ART/'bilin18_query_punctuation_copy_retest_native_cache.pt',weights_only=False);rows=[r for r in saved['rows'] if r['family'] in ('binding_new','copying_new')];by={r['id']:r for r in rows};cache=saved['cache'];native=saved['native'];selected={}
 for fam in ('binding','copying'):
  names=list(discovery['summary'][fam+'_old_paren']);scores={name:sum(discovery['summary'][fam+'_old_'+d][name]['phase_residual_ratio'] for d in ('paren','comma'))/2 for name in names if name!='self'};selected[fam]=sorted(scores,key=scores.get)[:2]
 modes=['self','same_pair','other_pair','union'];
 cfg=dict(experiment='H574',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),head=[14,4],selected=selected,modes=modes,selection='Two writes with lowest mean phase residual ratio across parentheses/commas in H573 old template, per task. Locked before new-template clamp results.',design='All new-template binding/copying delimiter cases from H572. New interventions on previously characterized prompts, not an independent lexical panel. Reconstruct final-position L14 pre-attention residual and raw first query using scaled differences of prior attention/MLP writes plus exact RMS denominator correction. Test every single prior write clamped to base final-position value; other computation live.',scope='Descriptive decomposition and causal write clamps are distinct. Only final-position writes changed; source-prefix keys unaffected. Discovery panel reused; no unique mediator or semantic-label claim.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;l,h=14,4;attn=model.blocks[l].attn;captures={};buckets=defaultdict(list)
 for r in rows:buckets[len(r['ids'])].append(r)
 for T,rr in buckets.items():
  for st in range(0,len(rr),16):
   bb=rr[st:st+16];cap={};handles=[]
   for layer,block in enumerate(model.blocks[:14]):
    def ah(m,args,out,layer=layer):cap[layer,'attn']=out[0][:,-1].cpu()
    def mh(m,args,out,layer=layer):cap[layer,'mlp']=out[:,-1].cpu()
    handles.extend([block.attn.register_forward_hook(ah),block.mlp.register_forward_hook(mh)])
   def pre(m,args):cap['r14']=(m.lambdas[0]*args[0][:,-1]+m.lambdas[1]*args[2][:,-1]).cpu()
   handles.append(model.blocks[14].register_forward_pre_hook(pre))
   try:run(model,torch.tensor([r['ids'] for r in bb],device='cuda'))
   finally:
    for handle in handles:handle.remove()
   for ix,r in enumerate(bb):captures[r['id']]={k:v[ix].clone() for k,v in cap.items()}
 result=dict(config=cfg,decomposition=[],interventions=[],status='running');reserrors=[];qerrors=[];jobs=defaultdict(list);gains={}
 for layer in range(14):
  gain=1.
  for future in range(layer+1,15):gain*=float(model.blocks[future].lambdas[0])
  gains[layer]=gain
 phase={(g['id']):g['predicted_coefficient'] for g in old['geometry'] if g['layer']==14};actual={(g['id']):g['actual_coefficient'] for g in old['geometry'] if g['layer']==14}
 for r in rows:
  if r['gap']=='base':continue
  c=captures[r['id']];b=captures[r['baseid']+'-base'];x0=b['r14'].cuda();x1=c['r14'].cuda();eps=torch.finfo(x0.dtype).eps;s0=(x0.square().mean()+eps).sqrt();s1=(x1.square().mean()+eps).sqrt();qweight=attn.c_q.weight[h*D:(h+1)*D];dq=F.linear(F.rms_norm(x1,(x1.numel(),))-F.rms_norm(x0,(x0.numel(),)),qweight);terms=[];writes=[]
  for layer in range(14):
   for kind in ('attn','mlp'):
    dr=gains[layer]*(c[layer,kind]-b[layer,kind]).cuda();writes.append(dr);terms.append((f'{kind}_{layer}',F.linear(dr/s0,qweight)))
  terms.append(('normalization',F.linear(x1*(1/s1-1/s0),qweight)));reserrors.append(float((sum(writes)-(x1-x0)).norm()/(x1-x0).norm()));qerrors.append(float((sum(q for n,q in terms)-dq).norm()/dq.norm()))
  for name,q in terms:result['decomposition'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],component=name,projection_share=float(q@dq/dq.square().sum()),relative_norm=float(q.norm()/dq.norm())))
  for mode in modes:jobs[len(r['ids']),mode,r['family']].append(r)
 torch.save(captures,ART/f'{STEM}_writes.pt');print('LOCKED',sum(map(len,jobs.values())),flush=True)
 for (T,mode,family),rr in jobs.items():
  for st in range(0,len(rr),16):
   bb=rr[st:st+16];handles=[];qcap={}
   task=family.split('_')[0];other='copying' if task=='binding' else 'binding';components=[] if mode=='self' else selected[task] if mode=='same_pair' else selected[other] if mode=='other_pair' else sorted(set(selected[task]+selected[other]))
   for component in components:
    kind,layer=component.split('_');layer=int(layer);don=torch.stack([captures[r['baseid']+'-base'][layer,kind] for r in bb]).cuda()
    if kind=='attn':
     def hook(m,args,out,don=don):y,bus=out;y=y.clone();y[:,-1]=don.to(y);return y,bus
     handles.append(model.blocks[layer].attn.register_forward_hook(hook))
    else:
     def hook(m,args,out,don=don):out=out.clone();out[:,-1]=don.to(out);return out
     handles.append(model.blocks[layer].mlp.register_forward_hook(hook))
   for n in ('c_q','c_q2'):
    def hook(m,args,out,n=n):qcap[n]=out[:,-1].clone()
    handles.append(getattr(attn,n).register_forward_hook(hook))
   try:zz,_=run(model,torch.tensor([r['ids'] for r in bb],device='cuda'))
   finally:
    for handle in handles:handle.remove()
   for ix,r in enumerate(bb):
    p=r['source_position'];cos,sin=attn.rotary(torch.zeros(1,T,H,D,device='cuda'));fs=[]
    for qn,kn in [('c_q','c_k'),('c_q2','c_k2')]:
     q=qcap[qn][ix].view(1,1,H,D).expand(1,T,H,D);k=cache[r['id']][l,kn][p].cuda().view(1,1,H,D).expand(1,T,H,D);q=model.tt.apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin)[0,-1,h];k=model.tt.apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin)[0,p,h];fs.append(float(q@k/D))
    coef=fs[0]*fs[1]*float(attn.lamb);result['interventions'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],mode=mode,coefficient=coef,native_coefficient=actual[r['id']],phase_coefficient=phase[r['id']],margin_delta=measure(zz[ix],r['groups'])['margin']-native[r['id']]['margin']))
 result['summary']={}
 for fam in ('binding_new','copying_new'):
  for delim in ('paren','comma'):
   key=fam+'_'+delim;result['summary'][key]={};gg=[g for g in result['decomposition'] if g['family']==fam and g['gap'].endswith(delim)];cc=[c for c in result['interventions'] if c['family']==fam and c['gap'].endswith(delim)];den=np.mean([abs(c['native_coefficient']-c['phase_coefficient']) for c in cc if c['mode']=='self'])
   for mode in modes:
    vv=[c for c in cc if c['mode']==mode];dd=[g for g in gg if g['component']==mode];result['summary'][key][mode]=dict(phase_residual_ratio=float(np.mean([abs(c['coefficient']-c['phase_coefficient']) for c in vv])/den),margin_abs_change=float(np.mean([abs(c['margin_delta']) for c in vv])),mean_projection_share=float(np.mean([g['projection_share'] for g in dd])) if dd else None)
 result['audits']=dict(residual_reconstruction_max_relative_error=max(reserrors),query_reconstruction_max_relative_error=max(qerrors),self_margin_max_error=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='self'),self_coefficient_max_error=max(abs(c['coefficient']-c['native_coefficient']) for c in result['interventions'] if c['mode']=='self'),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['audits']),flush=True)
if __name__=='__main__':main()
