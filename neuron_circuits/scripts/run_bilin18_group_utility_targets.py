"""H1008 discovery-only downstream utility of group plus scalar fill."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_group_utility_targets'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','geometry':ART/'bilin18_multi_anchor_geometry_bases.pt','old':ART/'bilin18_context_training_bank_captures.pt','noun':ART/'bilin18_residual_behavior_probes_captures.pt','nounscreen':ART/'bilin18_residual_behavior_probes.json','natural':ART/'bilin18_natural_residual_discovery_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};bd=torch.load(paths['geometry'],weights_only=False);Q=bd['bases']['multi_anchor128'];mu=bd['centers']['multi_anchor128'];old=torch.load(paths['old'],weights_only=False);noun=torch.load(paths['noun'],weights_only=False);ns=json.loads(paths['nounscreen'].read_text());nat=torch.load(paths['natural'],weights_only=False);items=[];seen=set()
 for i in bd['task_training_indices']:
  e=old['examples'][i]
  if e['text'] in seen:continue
  seen.add(e['text']);items.append(dict(domain='old',unit=e['id'],source_index=i,ids=e['ids'],write=old['writes'][i][12].double(),native=old['native'][i]))
 assert len(items)==132
 for r in ns['records']:
  if r['family']!='noun_expectation' or r['split']!='discovery':continue
  for i in r['indices']:items.append(dict(domain='noun',unit=r['id'],source_index=i,ids=noun['examples'][i]['ids'],write=noun['writes'][i][12].double(),native=noun['native'][i]))
 for i in bd['natural_training_indices']:
  r=nat['metadata'][i];assert r['split']=='discovery';items.append(dict(domain='natural',unit=r['document_hash'],source_index=i,ids=nat['tokens'][r['document'],:r['position']+1].tolist(),write=nat['writes'][i].double(),native=None))
 assert len(items)==908
 groups=bd['group_indices'];members={'real':groups}
 for seed in [10081,10082]:
  perm=torch.randperm(128,generator=torch.Generator().manual_seed(seed));members[f'shuffle{seed}']={name:perm[indices].tolist() for name,indices in groups.items()}
 X=torch.stack([r['write'] for r in items]);C=(X-mu)@Q;energy=C.square();supports={};scalar=torch.zeros_like(C,dtype=torch.bool);scalar.scatter_(1,energy.topk(32,dim=1).indices,True);supports['scalar32']=scalar
 for panel,gg in members.items():
  for name,indices in gg.items():
   mask=torch.zeros_like(scalar);mask[:,indices]=True;score=energy.clone();score[:,indices]=-float('inf');mask.scatter_(1,score.topk(32-len(indices),dim=1).indices,True);assert bool((mask.sum(1)==32).all());supports[panel+'/'+name]=mask
 arms=['identity','dense128']+list(supports);cfg=dict(experiment='H1008',hashes=hashes,design='Discovery-only utility targets. Deduplicate132old training prompts,8noun discovery endpoints,768natural discovery states. Fixed multi-anchor dictionary and mean. Exactly32active coordinates: scalarTop32 or one forced anchor group plus remaining highest-energy coordinates. Seven real groups, two atom-permuted panels preserving sizes/overlap. Identity and dense128 references.',selection='Per-state lowest native-outputKL among scalar and seven panel choices is an in-sample oracle; it uses the native downstream output and is not a deployable encoder. All candidates saved for future training. No evaluation/confirmation states used.',metric='KL(native || candidate). No paired-edit optimization in this target collection; whole-state and paired validation required for later learned router.',limits='Only8nounstates, repeated task vocabulary,192natural training documents. Oracle includes scalar and therefore cannot be worse by construction; compare gain size against equal-choice shuffled panels. Group-plus-fill is hybrid sparsity, not pure group sparsity. No neural gate trained.')
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.save(dict(coordinates=C,supports=supports,memberships=members,metadata=[{k:r[k] for k in ['domain','unit','source_index']} for r in items]),ART/f'{S}_inputs.pt');input_hash=sha(ART/f'{S}_inputs.pt');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 actual=0;replay=0.;buckets=defaultdict(list)
 for i,r in enumerate(items):
  if r['native'] is None:buckets[len(r['ids'])].append(i)
 for length,ii in sorted(buckets.items()):
  for st in range(0,len(ii),24):
   inds=ii[st:st+24];y,cap=forward(model,torch.tensor([items[i]['ids'] for i in inds],device='cuda'),capture=True);actual+=len(inds)
   for k,i in enumerate(inds):items[i]['native']=y[k].double().cpu();replay=max(replay,float((cap[12][k].double()-items[i]['write']).norm()/items[i]['write'].norm()))
 jobs=defaultdict(list)
 for i,r in enumerate(items):
  for arm in arms:
   pred=r['write'] if arm=='identity' else mu+Q@C[i] if arm=='dense128' else mu+Q@(C[i]*supports[arm][i]);jobs[len(r['ids'])].append((i,arm,pred.float()))
 losses=torch.zeros(len(items),len(arms),dtype=torch.double);samples=[];counts=defaultdict(int);identity=0.
 for length,jj in sorted(jobs.items()):
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];y,_=forward(model,torch.tensor([items[t[0]]['ids'] for t in batch],device='cuda'),patch=(12,torch.stack([t[2] for t in batch])));actual+=len(batch)
   for k,(i,arm,pred) in enumerate(batch):
    out=y[k].double().cpu();base=items[i]['native'];ln=base.log_softmax(0);lp=out.log_softmax(0);loss=float((ln.exp()*(ln-lp)).sum());losses[i,arms.index(arm)]=loss
    if arm=='identity':identity=max(identity,float((out-base).abs().max()))
    key=items[i]['domain'],arm
    if counts[key]<1:samples.append(dict(index=i,arm=arm,output=out.float(),native=base.float(),KL=loss));counts[key]+=1
  print('DONE',length,actual,flush=True)
 summary=[];oracle={}
 for panel in members:
  indices=[arms.index('scalar32')]+[arms.index(panel+'/'+name) for name in groups];values,chosen=losses[:,indices].min(1);oracle[panel]=dict(KL=values,chosen=torch.tensor(indices)[chosen])
 for domain in ['old','noun','natural']:
  ii=[i for i,r in enumerate(items) if r['domain']==domain];base=float(losses[ii,arms.index('scalar32')].mean());dense=float(losses[ii,arms.index('dense128')].mean());entry=dict(domain=domain,n=len(ii),scalar_KL=base,dense_KL=dense,panels={})
  for panel,result in oracle.items():
   ll=result['KL'][ii];choice=result['chosen'][ii];entry['panels'][panel]=dict(oracle_KL=float(ll.mean()),relative_reduction=1-float(ll.mean())/base,non_scalar_fraction=float((choice!=arms.index('scalar32')).double().mean()),selection_counts={arms[k]:int((choice==k).sum()) for k in choice.unique().tolist()})
  summary.append(entry)
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),inputs_unchanged=sha(ART/f'{S}_inputs.pt')==input_hash,whole_model_forwards=actual,expected_forwards=768+len(items)*len(arms),identity_logits_max_abs=identity,natural_capture_relative_error=replay,finite=bool(torch.isfinite(losses).all()),evaluation_states_used=0,no_router_training=True)
 assert audits['hashes_unchanged'] and audits['inputs_unchanged'] and actual==audits['expected_forwards'] and identity<1e-4 and replay<1e-5 and audits['finite']
 torch.save(dict(arms=arms,KL=losses,oracle=oracle,samples=samples),ART/f'{S}_targets.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(summary,audits,flush=True)
if __name__=='__main__':main()
