"""H749: actual query/key node interventions, beyond a surgical edge coefficient."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_query_key_edit_subspaces import projected_edit
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_operation_state_interventions'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'captures':ART/'bilin18_fresh_wording_subspace_effects_captures.pt','results':ART/'bilin18_fresh_wording_subspace_effects.json','bases':ART/'bilin18_operation_nuisance_subspaces_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(v) for k,v in paths.items()}
 cfg={'experiment':'H749','hashes':hashes,'design':'All H747 structurally matched contexts, competent and diagnostic. Frozen rank4 original/pooled operation, identity/wording nuisance, random7421; full donor/native controls. Actual L13H0 normalized pre-RoPE q2 at final query and k2 at source are replaced jointly or separately by hooking their projection outputs. Preserve recipient raw projection norm, hence recipient normalized norm. All incident edges and both local/bus values respond naturally. Other heads/factors unchanged directly; downstream live. No absent-value surgery. Measure full-logit distribution, native intended margin and head change; compare against mode-matched full donor state intervention.', 'limitations':'Reuses H747 contexts. Intervention at projected Q/K states, not yet residual-stream/upstream circuit mediation. Whole attention-node change differs from single-edge bus-only intervention. No fitting, no task accuracy improvement criterion. Full donor normalized to recipient norm before insertion.'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cap=torch.load(paths['captures'],weights_only=False);old=json.loads(paths['results'].read_text());bases=torch.load(paths['bases'],weights_only=False)['bases'];meta={(r['id'],r['receiver']):r for r in old['records'] if r['scope']=='native'};states=cap['states'];examples=cap['examples'];jobs=[]
 scopes=['frozen_H742','operation_pooled','identity','wording','random_7421','full','native']
 for key,donor in cap['pairs']:
  r,d=states[key],states[donor]
  for scope in scopes:
   desired={}
   for c,u in [('q2','uq2'),('k2','uk2')]:
    if scope=='native':desired[c]=r[u]
    elif scope=='full':desired[c]=d[u]*(r[u].norm()/d[u].norm())
    else:desired[c]=projected_edit(r[u],d[u],bases[scope,c],True)
   for mode in ['query','key','joint']:jobs.append({'key':key,'donor':donor,'scope':scope,'mode':mode,'desired':desired})
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 attn=model.blocks[13].attn;buckets=defaultdict(list)
 for job in jobs:buckets[len(examples[job['key']]['ids'])].append(job)
 logits_cache={};heads={};normerrors=[];stateerrors=[];forwards=0
 for length,jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];hooks=[];headcap={}
   for c,projection in [('q2',attn.c_q2),('k2',attn.c_k2)]:
    def edit(m,args,out,c=c):
     y=out.clone()
     for i,job in enumerate(bb):
      if job['mode']!=('query' if c=='q2' else 'key') and job['mode']!='joint':continue
      pos=length-1 if c=='q2' else examples[job['key']]['source_position'];raw=out[i,pos,:128];target=job['desired'][c].to(raw);replacement=target*(raw.norm()/target.norm());y[i,pos,:128]=replacement
      normerrors.append(float((replacement.norm()-raw.norm()).abs()/raw.norm().clamp_min(1e-12)));got=F.rms_norm(replacement,(128,));stateerrors.append(float((got-target).norm()/target.norm().clamp_min(1e-12)))
     return y
    hooks.append(projection.register_forward_hook(edit))
   def capture(m,args):headcap['value']=args[0][:,-1,:128].detach().cpu()
   hooks.append(attn.c_proj.register_forward_pre_hook(capture))
   try:logits,_=run(model,torch.tensor([examples[j['key']]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb)
   for i,job in enumerate(bb):
    k=(*job['key'],job['scope'],job['mode']);logits_cache[k]=logits[i].cpu();heads[k]=headcap['value'][i]
 print('FORWARDS',forwards,flush=True)
 records=[];identity=[];modeidentity=[]
 for job in jobs:
  key=job['key'];scope=job['scope'];mode=job['mode'];m=meta[key];p=logits_cache[(*key,scope,mode)];n=cap['native_logits'][key];f=logits_cache[(*key,'full',mode)];e=examples[key];orient=1 if key[1]==0 else -1
  lp=F.log_softmax(p,dim=-1);ln=F.log_softmax(n,dim=-1);lf=F.log_softmax(f,dim=-1)
  margin=orient*(measure(p,e['groups'])['margin']-measure(n,e['groups'])['margin']);fullmargin=orient*(measure(f,e['groups'])['margin']-measure(n,e['groups'])['margin']);nh=heads[(*key,'native',mode)];ph=heads[(*key,scope,mode)];fh=heads[(*key,'full',mode)]
  cell='|'.join(m[k] for k in ['name_set','wording','format']);competent=all(old['competence'][cell+'|'+fam]['pass'] for fam in ['entity_binding','sequence_copy'])
  records.append({**{k:m[k] for k in ['id','receiver','name_set','wording','format','family','item']},'scope':scope,'mode':mode,'competent':competent,'margin_change':margin,'full_margin_change':fullmargin,'native_to_edit_KL':float((ln.exp()*(ln-lp)).sum()),'full_to_edit_KL':float((lf.exp()*(lf-lp)).sum()),'full_to_native_KL':float((lf.exp()*(lf-ln)).sum()),'head_error_squared':float((ph-fh).square().sum()),'full_head_change_squared':float((fh-nh).square().sum()),'intended_top1':measure(p,e['groups'])['top1'][key[1]]})
  if scope=='native':identity.append(float((p-n).abs().max()));modeidentity.append(float((p-logits_cache[(*key,'native','joint')]).abs().max()))
 groups=defaultdict(list)
 for r in records:groups[r['competent'],r['format'],r['mode'],r['scope']].append(r)
 summary=[]
 for (competent,fmt,mode,scope),rr in groups.items():
  den=sum(r['full_margin_change']**2 for r in rr);hden=sum(r['full_head_change_squared'] for r in rr)
  summary.append({'competent':competent,'format':fmt,'mode':mode,'scope':scope,'contexts':len(rr),'margin_relative_RMS':(sum((r['margin_change']-r['full_margin_change'])**2 for r in rr)/max(den,1e-30))**.5,'head_relative_RMS':(sum(r['head_error_squared'] for r in rr)/max(hden,1e-30))**.5,'mean_full_to_edit_KL':sum(r['full_to_edit_KL'] for r in rr)/len(rr),'mean_full_to_native_KL':sum(r['full_to_native_KL'] for r in rr)/len(rr),'mean_native_to_edit_KL':sum(r['native_to_edit_KL'] for r in rr)/len(rr),'full_margin_RMS':(den/len(rr))**.5,'intended_top1':sum(r['intended_top1'] for r in rr)/len(rr)})
 audits={'hashes_unchanged':{k:sha(v)==hashes[k] for k,v in paths.items()},'raw_norm_relative_error_max':max(normerrors),'normalized_state_relative_error_max':max(stateerrors),'native_logit_replay_max_abs':max(identity),'native_mode_replay_max_abs':max(modeidentity),'finite':all(torch.isfinite(v).all().item() for v in logits_cache.values()),'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(normerrors)<1e-5 and max(stateerrors)<1e-5 and max(identity)<1e-4 and max(modeidentity)<1e-4
 torch.save({'logits':logits_cache,'heads':heads},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in summary:
  if s['competent']:print(s,flush=True)
if __name__=='__main__':main()
