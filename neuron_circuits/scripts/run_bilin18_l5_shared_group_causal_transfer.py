"""H758 systematic earlier-output swaps with full selected Q2/K2 node clamps."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_l5_shared_group_memberships import select_pairs,union_pairs
from run_bilin18_verified_head_sparse import SignedBSF
from run_bilin18_l5_natural_featurizer_baselines import reconstruct
STEM='bilin18_l5_shared_group_causal_transfer'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'memberships':ART/'bilin18_l5_shared_group_memberships_captures.pt','group_implementation':ROOT/'scripts/run_bilin18_l5_shared_group_memberships.py','featurizers':ART/'bilin18_l5_sparse_featurizers_captures.pt','natural_fits':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','native_heads':ART/'bilin18_operation_l5_head_mediation_captures.pt','upstream_capture':ART/'bilin18_operation_upstream_mediation_captures.pt','capture':ART/'bilin18_fresh_wording_subspace_effects_captures.pt','results':ART/'bilin18_fresh_wording_subspace_effects.json','bases':ART/'bilin18_operation_nuisance_subspaces_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(v) for k,v in paths.items()}
 cfg={'experiment':'H758','hashes':hashes,'design':'Frozen H757 learned correlation-initialized overlapping groups, all3 scalar dictionaries and3 degree-preserving rewires per dictionary. Compare scalar8 and full-head references. Decode two selected groups through exact union of shared atoms, same select_pairs gate as training evaluation; no task optimization. Reconstruct endpoints separately then add F(donor)-F(receiver) to native L5H3/H5 at both selected positions. Compare total, full-QK-node-dependent and compact-operation-dependent effects as H756. All240 H747 contexts; native competence retained.','limitations':'Reused task panel. Fixed dictionary/energy-based grouping, not joint learned gates. Degree controls preserve membership counts but not anchor designation. Mean errors across seeds not independent statistical replication. Spherical downstream clamp includes complement correction.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cap=torch.load(paths['capture'],weights_only=False);old=json.loads(paths['results'].read_text());bases=torch.load(paths['bases'],weights_only=False)['bases'];examples=cap['examples'];meta={(r['id'],r['receiver']):r for r in old['records'] if r['scope']=='native'}
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 previous=torch.load(paths['upstream_capture'],weights_only=False);native_outputs=previous['native_outputs'];rawstates=previous['rawstates'];forwards=0
 native_heads=torch.load(paths['native_heads'],weights_only=False)['native_heads'];natural_fits=torch.load(paths['natural_fits'],weights_only=False)['fits'];learned=torch.load(paths['featurizers'],weights_only=False)['states'];memberships=torch.load(paths['memberships'],weights_only=False)['memberships'];featurizers=['full']+[f'scalar_{seed}' for seed in [7551,7552,7553]]+[f'group_{seed}' for seed in [7551,7552,7553]]+[f'null_{seed}_{ns}' for seed in [7551,7552,7553] for ns in [1,2,3]];reconstructed={};keys=list(native_heads);unionerrors=[];occupancy=[]
 for h in [3,5]:
  site=f'L5H{h}';fit=natural_fits[site];raw=torch.stack([native_heads[k][:,h*128:(h+1)*128] for k in keys]).double();flat=raw.reshape(-1,128)
  for variant in featurizers:
   if variant=='full':rec=flat
   else:
    parts=variant.split('_');kind,seed=parts[0],int(parts[1]);state=learned[site,1,seed];m=SignedBSF(1);m.load_state_dict(state['model']);m.eval();x=((flat-fit['mean'])@fit['write_pca']['encode']/state['input_scale']).float()
    if kind=='scalar':decoded=m(x)
    else:
     M=memberships[site,seed,'learned_raw' if kind=='group' else 'learned_raw_rewired',None if kind=='group' else int(parts[2])];z=m.encoder(x);pairs=select_pairs(z.square(),M);q=union_pairs(M,pairs);g=torch.zeros(len(x),128).scatter_(1,pairs,1);unionerrors.append(float((q-(g@M>0).float()).abs().max()));assert bool((q.sum(-1)<=8).all()) and bool((g.sum(-1)==2).all());decoded=(z*q)@m.decoder+m.bias;occupancy.append({'head':h,'variant':variant,'mean_unique_atoms':float(q.sum(-1).mean()),'overlap_fraction':float((q.sum(-1)<8).float().mean())})
    rec=fit['mean']+(decoded.double()*state['input_scale'])@fit['write_pca']['decode']
   rec=rec.view_as(raw)
   for i,key in enumerate(keys):reconstructed[key,h,variant]=rec[i].float()
 jobs=[];sites=[(h,v) for h in [3,5] for v in featurizers];modes=['free','full','frozen_H742']
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
      pos=[length-1,examples[j['key']]['source_position']];y[i,pos,sl]+=(reconstructed[j['donor'],head,kind]-reconstructed[j['key'],head,kind]).to(y)
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
 audits={'exact_union_max_abs_error':max(unionerrors),'hashes_unchanged':{k:sha(v)==hashes[k] for k,v in paths.items()},'baseline_and_self_swap_logit_error_max':max(controlerrors),'raw_full_node_clamp_max_abs':max(clamp_errors),'compact_coordinate_error_max_abs':max(compact_errors),'normalized_norm_error_max_abs':max(norm_errors),'complement_gain_range':[min(gains),max(gains)],'finite':finite,'paired_interventions':len(paired),'whole_model_forwards':forwards,'test_evaluations':0};print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and finite and max(controlerrors)<1e-4 and max(clamp_errors)==0 and max(compact_errors)<1e-5 and max(norm_errors)<1e-5 and len(paired)==7680 and max(unionerrors)==0
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'task_occupancy':occupancy,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n')
 for s in summary:
  if s['competent'] and s['clamp'] not in ['free','full']:print(s,flush=True)
if __name__=='__main__':main()
