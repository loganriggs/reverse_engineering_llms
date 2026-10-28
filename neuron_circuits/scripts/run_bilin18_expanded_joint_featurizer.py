"""H989 data-coverage comparison at fixed MLP12, with frozen H984 baselines."""
from run_bilin18_joint_change_featurizer import ROOT,ART,sha
from feature_structures.models import SharedGroups
from pathlib import Path
from torch import nn
import torch,json,time
S='bilin18_expanded_joint_featurizer'
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 six=['agreement','capital_recall','entity_binding','sentiment','negation','pronoun_gender'];families=six+['antonym'];seeds=[9831,9832];objectives=['reconstruction','paired_changes']
 files=[ART/f'joint_change_featurizer/{k}_{o}_{s}.pt' for k in ['block','shared_fixed'] for o in objectives for s in seeds]
 paths=[Path(__file__),ROOT/'scripts/feature_structures/models.py',ART/'bilin18_context_training_bank.json',ART/'bilin18_context_training_bank_captures.pt',ART/'bilin18_context_training_bank_training_eligibility.json',ART/'bilin18_expanded_task_bank.json',ART/'bilin18_task_bank_causal_sites_captures.pt']+files;hashes={str(p):sha(p) for p in paths}
 bank=json.loads(paths[2].read_text());data=torch.load(paths[3],weights_only=False,map_location='cpu');eligible=set(json.loads(paths[4].read_text())['training_indices']);oldbank=json.loads(paths[5].read_text());old=torch.load(paths[6],weights_only=False,map_location='cpu')['writes'];gates={(r['family'],r['context'],r['split']):r['pass_gate'] for r in bank['summary']};oldgates={(r['family'],r['split']):r['pass_gate'] for r in oldbank['summary']};meta=[];pairs=[];xs=[]
 for source,rows,writes in [('context',bank['records'],data['writes']),('holdout',oldbank['records'],old)]:
  for r in rows:
   if not (r['family'] in families if source=='context' else r['family'] in ['sequence_copy','spatial_inverse']):continue
   ii=[]
   for end,i in enumerate(r['indices']):
    ii.append(len(meta));xs.append(writes[i][12]);meta.append(dict(source=source,index=i,family=r['family'],split=r['split'],context=r.get('context',-1),pair=r['id'],endpoint=end,text=r['prompts'][end],train_eligible=source=='context' and i in eligible))
   pairs.append(dict(indices=ii,source=source,family=r['family'],split=r['split'],context=r.get('context',-1),native_pass=gates[r['family'],r['context'],r['split']] if source=='context' else oldgates[r['family'],r['split']]))
 X=torch.stack(xs).float().cuda();dest=ART/'expanded_joint_featurizer';dest.mkdir(exist_ok=True)
 cfg=dict(experiment='H989',hashes=hashes,layer=12,seeds=seeds,steps=600,regimes=['frozen_small6','expanded6','expanded7'],objectives=objectives,training_slots={'expanded6':144,'expanded7':152},design='Frozen H984 small6 baselines; fresh expanded six-family and seven-family fits. Same128 atoms/32groups, initial parameters, Adam learning rate and600steps. Full-batch fits expose more states per step; not matched FLOPs. Equal-family reconstruction and normalized pair-change means avoid family-count weighting changes. No validation selection.',limits='Country-language excluded atMLP12. New context panel plus old whole-task holdouts reused for all models. Changes include training distribution, train-derived scaling and sample count, not a pure sample-size randomized experiment. No causal model forwards.')
 (ROOT/'configs'/f'{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');records=[];audits=[]
 for kind in ['block','shared_fixed']:
  for seed in seeds:
   for objective in objectives:
    oldstate=torch.load(ART/f'joint_change_featurizer/{kind}_{objective}_{seed}.pt',weights_only=False,map_location='cpu');M=oldstate['membership'].cuda()
    for regime in cfg['regimes']:
     torch.manual_seed(seed);model=SharedGroups(1152,128,32,8 if kind=='block' else 4,4 if kind=='block' else 8).cuda();model.membership_logits.requires_grad_(False);head=nn.Linear(64,6).cuda();history=[]
     if regime=='frozen_small6':
      model.load_state_dict(oldstate['model']);center=oldstate['center'].cuda();scale=oldstate['scale'].cuda();train=[]
     else:
      train=[i for i,r in enumerate(meta) if r['train_eligible'] and (regime=='expanded7' or r['family']!='antonym')];assert len(train)==cfg['training_slots'][regime];center=X[train].mean(0);scale=(X[train]-center).square().mean().sqrt()
      traintexts={meta[i]['text'] for i in train};assert not any(r['text'] in traintexts for r in meta if r['source']=='holdout' or r['split']!='discovery')
     Z=(X-center)/scale
     if train:
      index={i:k for k,i in enumerate(train)};tp=[p for p in pairs if p['indices'][0] in index];u=torch.tensor([index[p['indices'][0]] for p in tp],device='cuda');v=torch.tensor([index[p['indices'][1]] for p in tp],device='cuda');family_names=sorted({meta[i]['family'] for i in train});state_masks=[torch.tensor([meta[i]['family']==f for i in train],device='cuda') for f in family_names];pair_masks=[torch.tensor([p['family']==f for p in tp],device='cuda') for f in family_names];target=Z[train][v]-Z[train][u];energy=[target[mask].square().sum(1).mean() for mask in pair_masks];opt=torch.optim.Adam(model.parameters(),lr=.001)
      for step in range(600):
       out=model(Z[train],membership=M);pred=out['reconstruction'];recon=torch.stack([(pred[mask]-Z[train][mask]).square().mean() for mask in state_masks]).mean();delta_error=(pred[v]-pred[u]-target).square().sum(1);change=torch.stack([delta_error[mask].mean()/en for mask,en in zip(pair_masks,energy)]).mean();loss=recon+(.1*change if objective=='paired_changes' else 0);opt.zero_grad();loss.backward();assert torch.isfinite(loss);opt.step()
       if step%100==0:history.append(dict(step=step,reconstruction=float(recon.detach()),change=float(change.detach())))
     with torch.no_grad():
      chunks=[model(z,membership=M) for z in Z.split(128)];pred=torch.cat([r['reconstruction'] for r in chunks])*scale+center;codes=torch.cat([r['code'] for r in chunks]);active=torch.cat([r['support'].sum(1) for r in chunks]);assert torch.isfinite(pred).all();assert all(torch.equal(r['support'],(r['gates']@M>0).float()) for r in chunks)
      for source,f,context,split in sorted({(p['source'],p['family'],p['context'],p['split']) for p in pairs}):
       pp=[p for p in pairs if (p['source'],p['family'],p['context'],p['split'])==(source,f,context,split)];a=[p['indices'][0] for p in pp];b=[p['indices'][1] for p in pp];ix=a+b;delta=X[b]-X[a];error=pred[b]-pred[a]-delta
       records.append(dict(kind=kind,seed=seed,objective=objective,regime=regime,source=source,family=f,context=context,split=split,n=len(pp),native_pass=pp[0]['native_pass'],delta_error_ratio=float(error.square().sum()/delta.square().sum()),active_atoms=float(active[ix].mean())))
      replay=0.
      if regime=='frozen_small6':
       sourcebank=torch.load(ART/'bilin18_task_bank_causal_sites_captures.pt',weights_only=False,map_location='cpu')['writes'];ox=torch.stack([sourcebank[r['index']][12] for r in oldstate['metadata']]).cuda();oy=model((ox-center)/scale,membership=M)['reconstruction'].cpu();replay=float((oy-oldstate['reconstruction']).abs().max());assert replay<1e-5
      name=f'{kind}_{objective}_{regime}_{seed}';torch.save(dict(model=model.state_dict(),membership=M.cpu(),center=center.cpu(),scale=scale.cpu(),metadata=meta,pairs=pairs,training_indices=train,predicted_writes=pred.cpu(),code=codes.cpu(),history=history),dest/f'{name}.pt');audits.append(dict(name=name,training_slots=len(train),unique_training_prompts=len({meta[i]['text'] for i in train}),old_model_replay_max_error=replay,union_exact=True));print('DONE',name,flush=True)
 assert all(sha(p)==hashes[str(p)] for p in paths)
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,audits=audits,hashes_unchanged=True,whole_model_forwards=0,elapsed_seconds=time.time()-start),indent=2)+'\n');print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
