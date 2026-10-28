"""H775 frozen rank8 direct-KL and reconstruction fits on diverse causal edits."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_l5_direct_kl_rank8 import RankEight
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_direct_kl_causal_transfer'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'fits':ART/'bilin18_l5_kl_training_coverage_fits.pt','base':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','diverse':ART/'bilin18_fresh_head_tasks_captures.pt','diverse_heads':ART/'bilin18_l5_diverse_featurizer_transfer_captures.pt','diverse_results':ART/'bilin18_l5_diverse_featurizer_transfer.json','copy':ART/'bilin18_l5_fresh_copy_positions_captures.pt','copy_results':ART/'bilin18_l5_fresh_copy_positions.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};fit=torch.load(paths['fits'],weights_only=False);base=torch.load(paths['base'],weights_only=False)['fits']['L5H5'];states={n:s for n,s in fit['states'].items() if n=='initial' or '_n1024_' in n};cfg={'experiment':'H775','hashes':hashes,'variants':['full']+list(states),'design':'All3 frozen larger-pool directKL fits, all3 matched reconstruction fits, original rank8 PCA and full edit. Diverse five-family/two-format panel query-only; four-format copy panel query/source/both. Add F(donor)-F(receiver) to native L5H5. Reciprocal answer changes, nuisance and identity. Native competence and full effectRMS>=.25 gates for relative margin error. Also KL(full edit||compressed edit). No refitting or seed selection.','limitations':'Previously inspected research panels; many diverse task/site effects small, some native tasks fail competence. Rank8 linear representations, no sparse/group/manifold verification. Neither native-to-edit KL nor small error on tiny interventions establishes task causality.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];summary=[];identity=[];vectors=[];replay=[];initialreplay=[];forwards=0;competences={};full_logits={};E=base['write_pca']['encode'].float();B=base['write_pca']['decode'].float();mu=base['mean'].float();scale=fit['scale']
 for panelname in ['diverse','copy']:
  cap=torch.load(paths[panelname],weights_only=False);old=json.loads(paths[panelname+'_results'].read_text());competence=old['native_competence'];competences[panelname]=competence;examples=cap['examples'];native=cap['native_logits'];heads=(torch.load(paths['diverse_heads'],weights_only=False)['native_heads'] if panelname=='diverse' else cap['native_heads']);keys=list(heads);raw=torch.stack([(heads[k][640:768][None] if panelname=='diverse' else heads[k][:,640:768]) for k in keys]).float();npos=raw.shape[1];flat=raw.reshape(-1,128);recon={'full':raw}
  for name,state in states.items():net=RankEight();net.load_state_dict(state);recon[name]=(net((flat-mu)@E/scale)*scale@B+mu).reshape(len(keys),npos,128)
  keyindex={k:i for i,k in enumerate(keys)};jobs=[];positions=['query'] if panelname=='diverse' else ['query','source','both']
  for rid in sorted({k[0] for k in examples}):
   for rv,dv,mode in [(0,1,'task'),(1,0,'task'),(0,2,'nuisance'),(0,0,'identity')]:
    for name in recon:
     for pos in positions:jobs.append({'key':(rid,rv),'donor':(rid,dv),'mode':mode,'variant':name,'position':pos})
  buckets=defaultdict(list)
  for job in jobs:buckets[len(examples[job['key']]['ids'])].append(job)
  oldindex={(r['id'],r['receiver'],r['donor'],r.get('position','query'),r['variant']):r for r in old['records'] if r['head']==5}
  for length,jj in buckets.items():
   jj.sort(key=lambda q:q['variant']!='full')
   for lo in range(0,len(jj),12):
    bb=jj[lo:lo+12];patch=torch.stack([recon[q['variant']][keyindex[q['donor']]]-recon[q['variant']][keyindex[q['key']]] for q in bb]).cuda()
    def edit(m,args):
     y=args[0].clone()
     for i,q in enumerate(bb):
      indices=[0,1] if q['position']=='both' else [0 if q['position']=='query' else 1]
      for pi in indices:
       pos=length-1 if pi==0 else examples[q['key']]['source_position'];y[i,pos,640:768]+=patch[i,pi].to(y)
       if q['variant']=='full':vectors.append(float((y[i,pos,640:768].cpu()-raw[keyindex[q['donor']],pi]).abs().max()))
     return(y,)
    hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(edit)
    try:out,_=run(model,torch.tensor([examples[q['key']]['ids'] for q in bb],device='cuda'))
    finally:hook.remove()
    forwards+=len(bb)
    for i,q in enumerate(bb):
     key=q['key'];e=examples[key];logits=out[i].cpu();refkey=(panelname,key,q['donor'],q['position']);lp=logits.double().log_softmax(-1);ln=native[key].double().log_softmax(-1)
     if q['variant']=='full':full_logits[refkey]=lp
     lf=full_logits[refkey];delta=measure(logits,e['groups'])['margin']-measure(native[key],e['groups'])['margin'];r={'panel':panelname,'id':key[0],'receiver':key[1],'donor':q['donor'][1],'head':5,'variant':q['variant'],'position':q['position'],'mode':q['mode'],'family':e['family'],'format':e['split'],'competent':competence[e['family']][e['split']]['pass'],'margin_change':delta,'native_to_edit_KL':float((ln.exp()*(ln-lp)).sum()),'full_edit_to_compressed_KL':float((lf.exp()*(lf-lp)).sum())};records.append(r)
     if q['mode']=='identity':identity.append(float((logits-native[key]).abs().max()))
     if q['variant'] in ['full','initial']:
      oldname='full' if q['variant']=='full' else 'write_pca';err=abs(delta-oldindex[key[0],key[1],q['donor'][1],q['position'],oldname]['margin_change']);(replay if oldname=='full' else initialreplay).append(err)
  print('finished',panelname,'forwards',forwards,flush=True)
 refs={(r['panel'],r['id'],r['receiver'],r['donor'],r['position']):r for r in records if r['variant']=='full'};groups=defaultdict(list)
 for r in records:
  if r['mode']=='identity':continue
  f=refs[r['panel'],r['id'],r['receiver'],r['donor'],r['position']];groups[r['panel'],r['family'],r['format'],r['mode'],r['position'],r['variant'],r['competent']].append((r,f))
 for (panel,fam,fmt,mode,pos,var,competent),rr in groups.items():
  rms=(sum(f['margin_change']**2 for r,f in rr)/len(rr))**.5;err=(sum((r['margin_change']-f['margin_change'])**2 for r,f in rr)/len(rr))**.5;summary.append({'panel':panel,'family':fam,'format':fmt,'mode':mode,'position':pos,'variant':var,'competent':competent,'contexts':len(rr),'full_margin_RMS':rms,'absolute_RMS_error':err,'relative_RMS_error':err/rms if competent and rms>=.25 else None,'mean_full_edit_to_compressed_KL':sum(r['full_edit_to_compressed_KL'] for r,f in rr)/len(rr)})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'full_reference_margin_replay_max_abs':max(replay),'PCA_initial_margin_replay_max_abs':max(initialreplay),'identity_logit_replay_max_abs':max(identity),'full_donor_vector_error_max_abs':max(vectors),'finite':all(torch.isfinite(torch.tensor([r['margin_change'],r['full_edit_to_compressed_KL']])).all() for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(replay+initialreplay+identity+vectors)<1e-4
 (ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'native_competence':competences,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits,indent=2),flush=True)
if __name__=='__main__':main()
