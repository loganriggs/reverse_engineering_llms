"""H580: upstream writer clamps and isolated carrier-value mediation."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_carrier_upstream_writers'
TARGETS=[(4,0,1),(4,0,2),(7,8,3)]
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_suffix_token_values.json';old=json.loads(src.read_text());data=torch.load(ART/'bilin18_suffix_token_values_writes.pt',weights_only=False);rows=data['rows'];suffix=data['suffix'];by={r['id']:r for r in rows}
 cfg=dict(experiment='H580',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),targets=TARGETS,design='At each selected suffix position replace each earlier attention or MLP write with its aligned base value. Measure full downstream effect, then separately replay only the resulting local carrier value difference at native producer routing and fixed bus. Self and direct base-value controls. All H572 copying rows retained.',selection='Targets fixed by H579. Rank upstream writers on old template by reduction in local-value distance to base; assess new template without reselection.',precision='float32 computation with bf16-rounded loaded weights',scope='Conditional write replacements and isolated value-path interventions, not unique mediation or semantic-unit identification. KL is native-to-intervened; CE uses the intended single answer token.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;captures={};logits={};keys={};buckets=defaultdict(list)
 for r in rows:buckets[len(r['ids'])].append(r)
 for T,rr in buckets.items():
  for st in range(0,len(rr),16):
   bb=rr[st:st+16];cap={};handles=[]
   for l in range(7):
    def ah(m,args,out,l=l):cap[l,'attn']=out[0][:,T-4:T-1].cpu()
    def mh(m,args,out,l=l):cap[l,'mlp']=out[:,T-4:T-1].cpu()
    handles.extend([model.blocks[l].attn.register_forward_hook(ah),model.blocks[l].mlp.register_forward_hook(mh)])
   for l in (4,7):
    def vh(m,args,out,l=l):cap[l,'value']=out[:,T-4:T-1].view(len(bb),3,H,D).cpu()
    handles.append(model.blocks[l].attn.c_v.register_forward_hook(vh))
   try:zz,cc=run(model,torch.tensor([r['ids'] for r in bb],device='cuda'),capture=True)
   finally:
    for hook in handles:hook.remove()
   for ix,r in enumerate(bb):
    captures[r['id']]={k:v[ix].clone() for k,v in cap.items()};logits[r['id']]=zz[ix].cpu();keys[r['id']]={n:cc[14,n][ix,r['source_position']].clone() for n in ('c_k','c_k2')}
 phase={g['id']:g['predicted_coefficient'] for g in json.loads((ART/'bilin18_query_punctuation_copy_retest.json').read_text())['geometry'] if g['layer']==14};natcoef={g['id']:g['actual_coefficient'] for g in json.loads((ART/'bilin18_query_punctuation_copy_retest.json').read_text())['geometry'] if g['layer']==14}
 result=dict(config=cfg,interventions=[],summary={},status='running');jobs=defaultdict(list)
 for T,rr in buckets.items():
  edited=[r for r in rr if r['gap']!='base']
  for l,h,pos in TARGETS:
   for mode in ['self','base_value']+[f'{kind}_{u}' for u in range(l) for kind in ('attn','mlp')]:jobs[T,l,h,pos,mode]=edited
 print('LOCKED',sum(map(len,jobs.values())),'paired interventions',flush=True)
 def metrics(z,r):
  nz=logits[r['id']].to(z);lp=F.log_softmax(z,0);nlp=F.log_softmax(nz,0);answer=r['groups'][r['variant']];assert len(answer)==1
  return dict(margin_delta=measure(z,r['groups'])['margin']-measure(nz,r['groups'])['margin'],kl=float((nlp.exp()*(nlp-lp)).sum()),ce_delta=float(nlp[answer[0]]-lp[answer[0]]))
 for ji,((T,l,h,pos,mode),rr) in enumerate(jobs.items()):
  for st in range(0,len(rr),16):
   bb=rr[st:st+16];ids=torch.tensor([r['ids'] for r in bb],device='cuda');p=T-5+pos;handles=[];vc={}
   if mode not in ('self','base_value'):
    kind,u=mode.split('_');u=int(u);don=torch.stack([captures[r['baseid']+'-base'][u,kind][pos-1] for r in bb]).cuda()
    if kind=='attn':
     def wh(m,args,out):y,bus=out;y=y.clone();y[:,p]=don.to(y);return y,bus
    else:
     def wh(m,args,out):y=out.clone();y[:,p]=don.to(y);return y
    handles.append(getattr(model.blocks[u],kind).register_forward_hook(wh))
   def vh(m,args,out):vc['value']=out[:,p].view(len(bb),H,D)[:,h].clone()
   handles.append(model.blocks[l].attn.c_v.register_forward_hook(vh))
   try:full,_=run(model,ids)
   finally:
    for hook in handles:hook.remove()
   nv=torch.stack([captures[r['id']][l,'value'][pos-1,h] for r in bb]).cuda();bv=torch.stack([captures[r['baseid']+'-base'][l,'value'][pos-1,h] for r in bb]).cuda();iv=bv if mode=='base_value' else vc['value'];routing=torch.tensor([float(suffix[r['id'],l]['pattern'][pos]) for r in bb],device='cuda');delta=F.linear((iv-nv)*(1-model.blocks[l].attn.lamb)*routing[:,None],model.blocks[l].attn.c_proj.weight[:,h*D:(h+1)*D]);qc={};handles=[]
   for n in ('c_q','c_q2'):
    def qh(m,args,out,n=n):qc[n]=out[:,-1].clone()
    handles.append(getattr(model.blocks[14].attn,n).register_forward_hook(qh))
   try:mediated,_=run(model,ids,patch=(l,delta))
   finally:
    for hook in handles:hook.remove()
   target=model.blocks[14].attn;cos,sin=target.rotary(torch.zeros(1,T,H,D,device='cuda'))
   for ix,r in enumerate(bb):
    fs=[]
    for qn,kn in [('c_q','c_k'),('c_q2','c_k2')]:
     q=qc[qn][ix].view(1,1,H,D).expand(1,T,H,D);k=keys[r['id']][kn].cuda().view(1,1,H,D).expand(1,T,H,D);q=model.tt.apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin)[0,-1,4];k=model.tt.apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin)[0,r['source_position'],4];fs.append(float(q@k/D))
    result['interventions'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],layer=l,head=h,position=pos,mode=mode,value_residual_ratio=float((iv[ix]-bv[ix]).norm()/(nv[ix]-bv[ix]).norm()),value_delta_norm=float((iv[ix]-nv[ix]).norm()),coefficient=fs[0]*fs[1]*float(target.lamb),native_coefficient=natcoef[r['id']],phase_coefficient=phase[r['id']],full=None if mode=='base_value' else metrics(full[ix],r),mediated=metrics(mediated[ix],r)))
  if ji%24==0:print('PROGRESS',ji,len(jobs),flush=True)
 for fam in ('copying_old','copying_new'):
  for delim in ('paren','comma'):
   for l,h,pos in TARGETS:
    cc=[c for c in result['interventions'] if c['family']==fam and c['gap'].endswith(delim) and c['layer']==l and c['position']==pos];den=np.mean([abs(c['native_coefficient']-c['phase_coefficient']) for c in cc if c['mode']=='self']);key=f'{fam}_{delim}_L{l}H{h}_pos{pos}';result['summary'][key]={}
    for mode in sorted({c['mode'] for c in cc}):
     vv=[c for c in cc if c['mode']==mode];result['summary'][key][mode]=dict(value_residual_ratio=float(np.mean([c['value_residual_ratio'] for c in vv])),phase_residual_ratio=float(np.mean([abs(c['coefficient']-c['phase_coefficient']) for c in vv])/den),**{f'{path}_{metric}':float(np.mean([abs(c[path][metric]) if metric=='margin_delta' else c[path][metric] for c in vv])) for path in ('mediated','full') for metric in ('margin_delta','kl','ce_delta') if vv[0][path] is not None})
 prior={(c['id'],c['layer'],int(c['mode'][-1])):c for c in old['interventions'] if c['mode'].startswith('replace_pos')};selfs=[c for c in result['interventions'] if c['mode']=='self'];result['audits']=dict(self_full_margin_max_error=max(abs(c['full']['margin_delta']) for c in selfs),self_mediated_margin_max_error=max(abs(c['mediated']['margin_delta']) for c in selfs),base_value_replay_margin_max_error=max(abs(c['mediated']['margin_delta']-prior[c['id'],c['layer'],c['position']]['margin_delta']) for c in result['interventions'] if c['mode']=='base_value'),base_value_replay_coefficient_max_error=max(abs(c['coefficient']-prior[c['id'],c['layer'],c['position']]['coefficient']) for c in result['interventions'] if c['mode']=='base_value'),finite=all(np.isfinite(c['value_residual_ratio']) and np.isfinite(c['mediated']['kl']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(captures,ART/f'{STEM}_captures.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()
