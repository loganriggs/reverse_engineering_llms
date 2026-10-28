"""H751 systematic earlier-output swaps with full selected Q2/K2 node clamps."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_operation_upstream_mediation'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'capture':ART/'bilin18_fresh_wording_subspace_effects_captures.pt','results':ART/'bilin18_fresh_wording_subspace_effects.json','bases':ART/'bilin18_operation_nuisance_subspaces_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(v) for k,v in paths.items()}
 cfg={'experiment':'H751','hashes':hashes,'design':'Systematic L0..12 attention/MLP output swaps at both matched final-query and source-key positions between binding/copy H747 contexts. Run each swap with selected L13H0 raw Q2(final)/K2(source) free or clamped to native. All240 contexts retained, competence separately reported. Actual raw native states captured for exact clamp. Record intended-answer-margin changes, KL, and frozen H742 compact coordinate movement. No fitting. Baseline no-swap/free-or-clamp controls for every context; self-swap controls for each layer/component/mode on one fixed context.', 'limitations':'Full128D selected Q/K node clamp, not rank4-only mediation. Free-minus-clamped is an intervention dependence contrast, includes downstream interactions and is not a unique causal path fraction. Earlier output swap affects other heads and paths. Two positions swapped together. Data reused from H747.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cap=torch.load(paths['capture'],weights_only=False);old=json.loads(paths['results'].read_text());bases=torch.load(paths['bases'],weights_only=False)['bases'];examples=cap['examples'];meta={(r['id'],r['receiver']):r for r in old['records'] if r['scope']=='native'}
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native_outputs={};rawstates={};native_replay=[];forwards=0;buckets=defaultdict(list)
 for key,e in examples.items():buckets[len(e['ids'])].append(key)
 for length,keys in buckets.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];store={};hooks=[]
   def select(t):return torch.stack([t[i,[length-1,examples[k]['source_position']]] for i,k in enumerate(bb)]).detach().cpu()
   for layer in range(13):
    for kind in ['attn','mlp']:
     def capture(m,args,out,layer=layer,kind=kind):store[layer,kind]=select(out[0] if kind=='attn' else out)
     hooks.append(getattr(model.blocks[layer],kind).register_forward_hook(capture))
   for c in ['q2','k2']:
    def raw(m,args,out,c=c):store[c]=select(out[:,:,:128])
    hooks.append(getattr(model.blocks[13].attn,'c_'+c).register_forward_hook(raw))
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb)
   for i,key in enumerate(bb):
    native_replay.append(float((logits[i].cpu()-cap['native_logits'][key]).abs().max()));native_outputs[key]={k:v[i] for k,v in store.items() if isinstance(k,tuple)};rawstates[key]={'q2':store['q2'][i,0],'k2':store['k2'][i,1]}
 print('CAPTURED',forwards,flush=True)
 jobs=[]
 for key,donor in cap['pairs']:
  for layer in range(13):
   for kind in ['attn','mlp']:
    for clamp in [False,True]:jobs.append(dict(key=key,donor=donor,layer=layer,kind=kind,clamp=clamp,control='none'))
  for clamp in [False,True]:jobs.append(dict(key=key,donor=key,layer=-1,kind='baseline',clamp=clamp,control='baseline'))
 fixed=cap['pairs'][0][0]
 for layer in range(13):
  for kind in ['attn','mlp']:
   for clamp in [False,True]:jobs.append(dict(key=fixed,donor=fixed,layer=layer,kind=kind,clamp=clamp,control='self'))
 buckets=defaultdict(list)
 for j in jobs:buckets[j['layer'],j['kind'],j['clamp'],len(examples[j['key']]['ids'])].append(j)
 records=[];controlerrors=[];clamp_errors=[];finite=True
 for (layer,kind,clamp,length),jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];hooks=[];qk={}
   if layer>=0:
    def swap(m,args,out):
     x=out[0] if kind=='attn' else out;y=x.clone()
     for i,j in enumerate(bb):
      pos=[length-1,examples[j['key']]['source_position']];y[i,pos]=native_outputs[j['donor']][layer,kind].to(y)
     return (y,out[1]) if kind=='attn' else y
    hooks.append(getattr(model.blocks[layer],kind).register_forward_hook(swap))
   for c in ['q2','k2']:
    def node(m,args,out,c=c):
     y=out.clone() if clamp else out
     if clamp:
      for i,j in enumerate(bb):
       pos=length-1 if c=='q2' else examples[j['key']]['source_position'];y[i,pos,:128]=rawstates[j['key']][c].to(y)
     selected=torch.stack([y[i,length-1 if c=='q2' else examples[j['key']]['source_position'],:128] for i,j in enumerate(bb)])
     qk[c]=F.rms_norm(selected,(128,)).detach().cpu().double()
     if clamp:
      for i,j in enumerate(bb):clamp_errors.append(float((selected[i].cpu()-rawstates[j['key']][c]).abs().max()))
     return y
    hooks.append(getattr(model.blocks[13].attn,'c_'+c).register_forward_hook(node))
   try:logits,_=run(model,torch.tensor([examples[j['key']]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb);finite=finite and bool(torch.isfinite(logits).all())
   for i,j in enumerate(bb):
    key=j['key'];m=meta[key];e=examples[key];p=logits[i].cpu();n=cap['native_logits'][key];cell='|'.join(m[k] for k in ['name_set','wording','format']);competent=all(old['competence'][cell+'|'+fam]['pass'] for fam in ['entity_binding','sequence_copy']);margin=(measure(p,e['groups'])['margin']-measure(n,e['groups'])['margin'])*(1 if key[1]==0 else -1);lp=F.log_softmax(p,dim=-1);ln=F.log_softmax(n,dim=-1)
    rec={**{k:m[k] for k in ['id','receiver','name_set','wording','format','family','item']},'layer':layer,'kind':kind,'clamp':clamp,'control':j['control'],'competent':competent,'margin_change':margin,'KL':float((ln.exp()*(ln-lp)).sum())}
    for c,u in [('q2','uq2'),('k2','uk2')]:
     B=bases['frozen_H742',c];delta=(qk[c][i]-cap['states'][key][u])@B;target=(cap['states'][j['donor']][u]-cap['states'][key][u])@B;rec[c+'_alignment']=float(delta@target);rec[c+'_target_energy']=float(target.square().sum());rec[c+'_change_energy']=float(delta.square().sum())
    records.append(rec)
    if j['control']!='none':controlerrors.append(float((p-n).abs().max()))
 print('INTERVENED',forwards,flush=True)
 paired=defaultdict(dict)
 for r in records:
  if r['control']=='none':paired[r['id'],r['receiver'],r['layer'],r['kind']][r['clamp']]=r
 groups=defaultdict(list)
 for v in paired.values():
  f,c=v[False],v[True];groups[f['competent'],f['format'],f['layer'],f['kind']].append((f,c))
 summary=[]
 for (competent,fmt,layer,kind),rr in groups.items():
  den=sum(f['margin_change']**2 for f,c in rr);s={'competent':competent,'format':fmt,'layer':layer,'kind':kind,'contexts':len(rr),'free_margin_RMS':(den/len(rr))**.5,'clamped_margin_RMS':(sum(c['margin_change']**2 for f,c in rr)/len(rr))**.5,'dependence_margin_RMS':(sum((f['margin_change']-c['margin_change'])**2 for f,c in rr)/len(rr))**.5,'dependence_alignment':sum((f['margin_change']-c['margin_change'])*f['margin_change'] for f,c in rr)/max(den,1e-30),'mean_free_KL':sum(f['KL'] for f,c in rr)/len(rr),'mean_clamped_KL':sum(c['KL'] for f,c in rr)/len(rr)}
  for node in ['q2','k2']:s[node+'_coordinate_alignment']=sum(f[node+'_alignment'] for f,c in rr)/max(sum(f[node+'_target_energy'] for f,c in rr),1e-30)
  summary.append(s)
 audits={'hashes_unchanged':{k:sha(v)==hashes[k] for k,v in paths.items()},'native_replay_max_abs':max(native_replay),'baseline_and_self_swap_logit_error_max':max(controlerrors),'raw_node_clamp_max_abs':max(clamp_errors),'finite':finite,'paired_interventions':len(paired),'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and finite and max(native_replay)<1e-4 and max(controlerrors)<1e-4 and max(clamp_errors)==0 and len(paired)==6240
 torch.save({'native_outputs':native_outputs,'rawstates':rawstates},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in sorted([s for s in summary if s['competent']],key=lambda s:s['dependence_margin_RMS'],reverse=True)[:10]:print(s,flush=True)
if __name__=='__main__':main()
