"""H753 systematic earlier-output swaps with full selected Q2/K2 node clamps."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_operation_l5_head_mediation'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'upstream_capture':ART/'bilin18_operation_upstream_mediation_captures.pt','capture':ART/'bilin18_fresh_wording_subspace_effects_captures.pt','results':ART/'bilin18_fresh_wording_subspace_effects.json','bases':ART/'bilin18_operation_nuisance_subspaces_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(v) for k,v in paths.items()}
 cfg={'experiment':'H753','hashes':hashes,'design':'All nine L5 attention heads separately, plus all-heads replay (head9 sentinel). Swap head output coordinates before c_proj at matched destination/source positions between H747 operations. For each, run downstream Q2/K2 free, full native clamp, or rank4 original-operation/identity/wording/random7421 spherical clamps as H752. All240 contexts retained; competence reported separately. No fitting or head selection before sweep.','limitations':'Reused panel; head-coordinate swap affects all residual destinations of that head write. Downstream mediation contrast is not unique path decomposition. Both source and destination swapped. Spherical clamp rescales complement.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cap=torch.load(paths['capture'],weights_only=False);old=json.loads(paths['results'].read_text());bases=torch.load(paths['bases'],weights_only=False)['bases'];examples=cap['examples'];meta={(r['id'],r['receiver']):r for r in old['records'] if r['scope']=='native'}
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 previous=torch.load(paths['upstream_capture'],weights_only=False);native_outputs=previous['native_outputs'];rawstates=previous['rawstates'];forwards=0
 native_heads={};capture_replay=[];capture_groups=defaultdict(list)
 for key,e in examples.items():capture_groups[len(e['ids'])].append(key)
 for length,keys in capture_groups.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];saved={}
   def capture_heads(m,args):saved['heads']=torch.stack([args[0][i,[length-1,examples[k]['source_position']]] for i,k in enumerate(bb)]).detach().cpu()
   hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(capture_heads)
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for i,key in enumerate(bb):native_heads[key]=saved['heads'][i];capture_replay.append(float((logits[i].cpu()-cap['native_logits'][key]).abs().max()))
 print('CAPTURED',forwards,flush=True)
 jobs=[];sites=[(h,'attn') for h in range(10)];modes=['free','full','frozen_H742','identity','wording','random_7421']
 for key,donor in cap['pairs']:
  for head,kind in sites:
   for clamp in modes:jobs.append(dict(key=key,donor=donor,head=head,kind=kind,clamp=clamp,control='none'))
  for clamp in modes:jobs.append(dict(key=key,donor=key,head=-1,kind='baseline',clamp=clamp,control='baseline'))
 fixed=cap['pairs'][0][0]
 for head,kind in sites:
  for clamp in modes:jobs.append(dict(key=fixed,donor=fixed,head=head,kind=kind,clamp=clamp,control='self'))
 buckets=defaultdict(list)
 for j in jobs:buckets[j['head'],j['kind'],j['clamp'],len(examples[j['key']]['ids'])].append(j)
 records=[];controlerrors=[];clamp_errors=[];compact_errors=[];norm_errors=[];gains=[];finite=True
 for (head,kind,clamp,length),jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];hooks=[];qk={}
   if head>=0:
    def swap(m,args):
     y=args[0].clone();sl=slice(head*128,(head+1)*128) if head<9 else slice(None)
     for i,j in enumerate(bb):
      pos=[length-1,examples[j['key']]['source_position']];y[i,pos,sl]=native_heads[j['donor']][:,sl].to(y)
     return (y,)
    hooks.append(model.blocks[5].attn.c_proj.register_forward_pre_hook(swap))
   for c in ['q2','k2']:
    def node(m,args,out,c=c):
     y=out.clone() if clamp!='free' else out
     if clamp!='free':
      for i,j in enumerate(bb):
       pos=length-1 if c=='q2' else examples[j['key']]['source_position'];native=rawstates[j['key']][c].to(y)
       if clamp=='full':y[i,pos,:128]=native
       else:
        B=bases[clamp,c].to(device=y.device);current=F.rms_norm(y[i,pos,:128],(128,)).double();target=F.rms_norm(native,(128,)).double();parallel=(target@B)@B.T;orth=current-(current@B)@B.T;nativeorth=target-(target@B)@B.T;assert orth.norm()>1e-8;gain=nativeorth.norm()/orth.norm();desired=parallel+gain*orth;replacement=(desired*(native.double().norm()/target.norm())).to(y);y[i,pos,:128]=replacement;actual=F.rms_norm(replacement,(128,)).double();compact_errors.append(float(((actual-target)@B).abs().max()));norm_errors.append(float((actual.norm()-target.norm()).abs()));gains.append(float(gain))
     selected=torch.stack([y[i,length-1 if c=='q2' else examples[j['key']]['source_position'],:128] for i,j in enumerate(bb)])
     qk[c]=F.rms_norm(selected,(128,)).detach().cpu().double()
     if clamp=='full':
      for i,j in enumerate(bb):clamp_errors.append(float((selected[i].cpu()-rawstates[j['key']][c]).abs().max()))
     return y
    hooks.append(getattr(model.blocks[13].attn,'c_'+c).register_forward_hook(node))
   try:logits,_=run(model,torch.tensor([examples[j['key']]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb);finite=finite and bool(torch.isfinite(logits).all())
   for i,j in enumerate(bb):
    key=j['key'];m=meta[key];e=examples[key];p=logits[i].cpu();n=cap['native_logits'][key];cell='|'.join(m[k] for k in ['name_set','wording','format']);competent=all(old['competence'][cell+'|'+fam]['pass'] for fam in ['entity_binding','sequence_copy']);margin=(measure(p,e['groups'])['margin']-measure(n,e['groups'])['margin'])*(1 if key[1]==0 else -1);lp=F.log_softmax(p,dim=-1);ln=F.log_softmax(n,dim=-1)
    rec={**{k:m[k] for k in ['id','receiver','name_set','wording','format','family','item']},'head':head,'kind':kind,'clamp':clamp,'control':j['control'],'competent':competent,'margin_change':margin,'KL':float((ln.exp()*(ln-lp)).sum())}
    for c,u in [('q2','uq2'),('k2','uk2')]:
     B=bases['frozen_H742',c];delta=(qk[c][i]-cap['states'][key][u])@B;target=(cap['states'][j['donor']][u]-cap['states'][key][u])@B;rec[c+'_alignment']=float(delta@target);rec[c+'_target_energy']=float(target.square().sum());rec[c+'_change_energy']=float(delta.square().sum())
    records.append(rec)
    if j['control']!='none':controlerrors.append(float((p-n).abs().max()))
 print('INTERVENED',forwards,flush=True)
 paired=defaultdict(dict)
 for r in records:
  if r['control']=='none':paired[r['id'],r['receiver'],r['head'],r['kind']][r['clamp']]=r
 groups=defaultdict(list)
 for v in paired.values():
  f=v['free'];groups[f['competent'],f['format'],f['head'],f['kind']].append(v)
 summary=[]
 for (competent,fmt,head,kind),vv in groups.items():
  for mode in modes:
   rr=[(v['free'],v[mode],v['full']) for v in vv];den=sum((f['margin_change']-full['margin_change'])**2 for f,c,full in rr);total=sum(f['margin_change']**2 for f,c,full in rr)
   s={'competent':competent,'format':fmt,'head':head,'kind':kind,'clamp':mode,'contexts':len(rr),'full_dependence_RMS':(den/len(rr))**.5,'compact_dependence_RMS':(sum((f['margin_change']-c['margin_change'])**2 for f,c,full in rr)/len(rr))**.5,'dependence_relative_error':(sum((c['margin_change']-full['margin_change'])**2 for f,c,full in rr)/max(den,1e-30))**.5,'dependence_alignment':sum((f['margin_change']-c['margin_change'])*(f['margin_change']-full['margin_change']) for f,c,full in rr)/max(den,1e-30),'free_margin_RMS':(total/len(rr))**.5}
   summary.append(s)
 audits={'hashes_unchanged':{k:sha(v)==hashes[k] for k,v in paths.items()},'capture_native_replay_max_abs':max(capture_replay),'baseline_and_self_swap_logit_error_max':max(controlerrors),'raw_full_node_clamp_max_abs':max(clamp_errors),'compact_coordinate_error_max_abs':max(compact_errors),'normalized_norm_error_max_abs':max(norm_errors),'complement_gain_range':[min(gains),max(gains)],'finite':finite,'paired_interventions':len(paired),'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and finite and max(controlerrors)<1e-4 and max(clamp_errors)==0 and max(compact_errors)<1e-5 and max(norm_errors)<1e-5 and len(paired)==2400 and max(capture_replay)<1e-4
 torch.save({'native_heads':native_heads},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in summary:
  if s['competent'] and s['clamp'] not in ['free','full']:print(s,flush=True)
if __name__=='__main__':main()
