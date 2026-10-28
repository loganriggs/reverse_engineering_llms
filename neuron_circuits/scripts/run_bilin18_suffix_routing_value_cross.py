"""H577: source-group decomposition and native deletion/base replacement of producer writes."""
import json,hashlib,time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_suffix_routing_value_cross'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_producer_source_groups.json';old=json.loads((ART/'bilin18_query_punctuation_copy_retest.json').read_text());saved=torch.load(ART/'bilin18_query_punctuation_copy_retest_native_cache.pt',weights_only=False);rows=[r for r in saved['rows'] if r['task']=='copying'];by={r['id']:r for r in rows};native=saved['native'];groups=['selected_name','earlier_names','other_prefix','insertion','suffix','final_token'];cfg=dict(experiment='H578',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),producers=[[4,0],[7,8]],target=[14,4],groups=groups,design='Cross native/base routing patterns and native/base mixed values at aligned suffix positions, excluding final token. All copying templates/delimiters and both producers retained; local suffix contribution only.',scope='Base replacement removes the contextual change; deletion removes the entire native contribution. They answer different questions. Coarse token groups do not identify semantics or unique causal paths.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;cache={};buckets=defaultdict(list)
 for r in rows:buckets[len(r['ids'])].append(r)
 for T,rr in buckets.items():
  for st in range(0,len(rr),16):
   bb=rr[st:st+16];cap={};handles=[]
   for l,h in cfg['producers']:
    for n in ('c_q','c_k','c_q2','c_k2','c_v'):
     def hook(m,args,out,l=l,n=n):cap[l,n]=out.cpu()
     handles.append(getattr(model.blocks[l].attn,n).register_forward_hook(hook))
    def pre(m,args,l=l):cap[l,'preproj']=args[0].cpu()
    handles.append(model.blocks[l].attn.c_proj.register_forward_pre_hook(pre))
   try:_,cc=run(model,torch.tensor([r['ids'] for r in bb],device='cuda'),capture=True)
   finally:
    for handle in handles:handle.remove()
   cc.update(cap)
   for ix,r in enumerate(bb):cache[r['id']]={k:v[ix].clone() for k,v in cc.items()}
 writes={};suffix={};buserrors=[];recon=[];result=dict(config=cfg,geometry=[],interventions=[],status='running');jobs=defaultdict(list)
 for r in rows:
  c=cache[r['id']];T=len(r['ids']);p=r['source_position'];extra=T-len(by[r['baseid']+'-base']['ids']);names={t for gg in r['groups'] for t in gg};parts=dict(selected_name=[p],earlier_names=[i for i in range(p) if r['ids'][i] in names],other_prefix=[i for i in range(p) if r['ids'][i] not in names],insertion=list(range(p+1,p+1+extra)),suffix=list(range(p+1+extra,T-1)),final_token=[T-1]);assert sorted(i for vv in parts.values() for i in vv)==list(range(T))
  for l,h in cfg['producers']:
   a=model.blocks[l].attn;raw={n:c[l,n].cuda().view(1,T,H,D) for n in ('c_q','c_k','c_q2','c_k2')};cos,sin=a.rotary(raw['c_q']);v={n:model.tt.apply_rotary_emb(F.rms_norm(x,(D,)),cos,sin)[0,:,h] for n,x in raw.items()};pattern=(v['c_k']@v['c_q'][-1]/D)*(v['c_k2']@v['c_q2'][-1]/D);values=(1-a.lamb)*c[l,'c_v'].cuda().view(T,H,D)[:,h]+a.lamb*c['bus'].cuda()[:,h];w=a.c_proj.weight[:,h*D:(h+1)*D];sourcewrites=F.linear(pattern[:,None]*values,w);ww={g:sourcewrites[ps].sum(0) for g,ps in parts.items()};actual=F.linear(c[l,'preproj'][-1,h*D:(h+1)*D].cuda(),w);recon.append(float((sum(ww.values())-actual).norm()/actual.norm()));writes[r['id'],l]={g:x.cpu() for g,x in ww.items()};ps=parts['suffix'];suffix[r['id'],l]=dict(pattern=pattern[ps].cpu(),values=values[ps].cpu(),bus=c['bus'][ps,h].clone(),tokens=[r['ids'][i] for i in ps])
 for r in rows:
  if r['gap']=='base':continue
  for l,h in cfg['producers']:
   a=suffix[r['id'],l];b=suffix[r['baseid']+'-base',l];assert a['tokens']==b['tokens'];buserrors.append(float((a['bus']-b['bus']).abs().max()));weight=model.blocks[l].attn.c_proj.weight[:,h*D:(h+1)*D];actual=writes[r['id'],l]['suffix'];vectors={}
   for route,aa in [('native',a),('base',b)]:
    for value,vv in [('native',a),('base',b)]:vectors[route+'_'+value]=F.linear((aa['pattern'][:,None]*vv['values']).sum(0).cuda(),weight).cpu()
   for mode,v in vectors.items():jobs[len(r['ids']),l].append((r,h,mode,v-actual))
   delta=vectors['native_native']-vectors['base_base'];routing=vectors['native_base']-vectors['base_base'];value=vectors['base_native']-vectors['base_base'];interaction=delta-routing-value
   result['geometry'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],layer=l,head=h,total_change_norm=float(delta.norm()),routing_relative_norm=float(routing.norm()/delta.norm()),value_relative_norm=float(value.norm()/delta.norm()),interaction_relative_norm=float(interaction.norm()/delta.norm()),routing_projection=float(routing@delta/delta.square().sum()),value_projection=float(value@delta/delta.square().sum()),interaction_projection=float(interaction@delta/delta.square().sum())))
 torch.save({'writes':writes,'rows':rows},ART/f'{STEM}_writes.pt');print('LOCKED',sum(map(len,jobs.values())),flush=True);target=model.blocks[14].attn;phase={g['id']:g['predicted_coefficient'] for g in old['geometry'] if g['layer']==14};nativecoef={g['id']:g['actual_coefficient'] for g in old['geometry'] if g['layer']==14}
 for (T,l),jj in jobs.items():
  for st in range(0,len(jj),24):
   bb=jj[st:st+24];qc={};handles=[]
   for n in ('c_q','c_q2'):
    def hook(m,args,out,n=n):qc[n]=out[:,-1].clone()
    handles.append(getattr(target,n).register_forward_hook(hook))
   try:zz,_=run(model,torch.tensor([r['ids'] for r,h,m,d in bb],device='cuda'),patch=(l,torch.stack([d for r,h,m,d in bb])))
   finally:
    for handle in handles:handle.remove()
   cos,sin=target.rotary(torch.zeros(1,T,H,D,device='cuda'))
   for ix,(r,h,mode,d) in enumerate(bb):
    fs=[];p=r['source_position']
    for qn,kn in [('c_q','c_k'),('c_q2','c_k2')]:
     q=qc[qn][ix].view(1,1,H,D).expand(1,T,H,D);k=cache[r['id']][14,kn][p].cuda().view(1,1,H,D).expand(1,T,H,D);q=model.tt.apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin)[0,-1,4];k=model.tt.apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin)[0,p,4];fs.append(float(q@k/D))
    co=fs[0]*fs[1]*float(target.lamb);result['interventions'].append(dict(id=r['id'],family=r['family'],gap=r['gap'],layer=l,head=h,mode=mode,coefficient=co,phase_coefficient=phase[r['id']],native_coefficient=nativecoef[r['id']],margin_delta=measure(zz[ix],r['groups'])['margin']-native[r['id']]['margin']))
 result['summary']={}
 for fam in ('copying_old','copying_new'):
  for delim in ('paren','comma'):
   for l,h in cfg['producers']:
    cc=[c for c in result['interventions'] if c['family']==fam and c['gap'].endswith(delim) and c['layer']==l];den=np.mean([abs(c['native_coefficient']-c['phase_coefficient']) for c in cc if c['mode']=='native_native']);key=f'{fam}_{delim}_L{l}H{h}';result['summary'][key]={}
    for mode in sorted({c['mode'] for c in cc}):
     vv=[c for c in cc if c['mode']==mode];result['summary'][key][mode]=dict(phase_residual_ratio=float(np.mean([abs(c['coefficient']-c['phase_coefficient']) for c in vv])/den),margin_abs_change=float(np.mean([abs(c['margin_delta']) for c in vv])))
 prior={(c['id'],c['layer']):c for c in json.loads(src.read_text())['interventions'] if c['mode']=='replace_suffix'}
 result['audits']=dict(group_reconstruction_max_relative_error=max(recon),suffix_tokens_aligned=True,suffix_bus_max_error=max(buserrors),base_suffix_replay_max_margin_error=max(abs(c['margin_delta']-prior[c['id'],c['layer']]['margin_delta']) for c in result['interventions'] if c['mode']=='base_base'),self_margin_max_error=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='native_native'),finite=all(np.isfinite(c['margin_delta']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['audits']),flush=True)
if __name__=='__main__':main()
