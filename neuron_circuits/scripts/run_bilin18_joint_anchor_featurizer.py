"""Small-data joint-featurizer pilot; no bootstrap-discovery claim."""
import hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from feature_structures.models import SharedGroups
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts/bilin18_tensor_decomposition'; S='bilin18_joint_anchor_featurizer'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths=[ART/'bilin18_expanded_task_bank.json',ART/'bilin18_task_bank_causal_sites_captures.pt',Path(__file__),ROOT/'scripts/feature_structures/models.py']; hashes={str(p):sha(p) for p in paths}
 bank=json.loads(paths[0].read_text()); cache=torch.load(paths[1],weights_only=False,map_location='cpu')['writes']
 anchors=['agreement','capital_recall','entity_binding','sentiment','negation','pronoun_gender'];holdouts=['sequence_copy','spatial_inverse']
 rows=[r for r in bank['records'] if r['family'] in anchors+holdouts and all(i in cache for i in r['indices'])]
 meta=[dict(family=r['family'],split=r['split'],pair=r['id'],endpoint=e,index=i,holdout=r['family'] in holdouts) for r in rows for e,i in enumerate(r['indices'])]
 train=torch.tensor([i for i,r in enumerate(meta) if not r['holdout'] and r['split']=='discovery']);assert len(train)==48
 assert not any(meta[i]['holdout'] for i in train.tolist())
 X=torch.stack([cache[r['index']][12] for r in meta]).float();center=X[train].mean(0);scale=(X[train]-center).square().mean().sqrt();X=((X-center)/scale).cuda();train=train.cuda()
 family=torch.tensor([anchors.index(r['family']) if r['family'] in anchors else 0 for r in meta],device='cuda'); y=torch.tensor([r['endpoint'] for r in meta],device='cuda').float()
 cfg=dict(experiment='H983',site='last-token MLP12 output, common to every task',anchors=anchors,whole_task_holdouts=holdouts,train_states=48,layer=12,seeds=[9831,9832],steps=600,atoms=128,groups=32,supervised_atoms=64,label_weight=.1,hashes=hashes,selection='Fixed final step; no validation checkpoint or hyperparameter selection.',limits='Tiny development-only pilot; group counts matched, active atoms approximately matched, registered parameters matched. Block groups use4 distinct atoms and8 gates; shared groups use8 atoms and4 gates, balanced degree2 fixed memberships. Shared membership learning deferred because previous collapse. No within-block orthogonality. Last64 atoms have no direct label readout but remain coupled via reconstruction and gates. No causal model intervention, no unsupervised task discovery. Output-write reconstruction differs from input-state featurization.')
 (ROOT/'configs'/f'{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 gate={(r['family'],r['split']):r['pass_gate'] for r in bank['summary']};records=[];audits=[];dest=ART/'joint_anchor_featurizer';dest.mkdir(exist_ok=True)
 for kind in ['block','shared_fixed']:
  for seed in cfg['seeds']:
   gen=torch.Generator().manual_seed(seed);M=torch.zeros(32,128)
   if kind=='block':M[torch.arange(32)[:,None],torch.arange(128).reshape(32,4)]=1
   else:
    for half in range(2):M[torch.arange(16)[:,None]+half*16,torch.randperm(128,generator=gen).reshape(16,8)]=1
   for objective in ['reconstruction','true_labels','shuffled_labels']:
    torch.manual_seed(seed);m=SharedGroups(1152,128,32,8 if kind=='block' else 4,4 if kind=='block' else 8).cuda();m.membership_logits.requires_grad_(False);head=nn.Linear(64,6).cuda();membership=M.cuda()
    # All three conditions share model initialization, stream, and label head.
    opt=torch.optim.Adam(list(m.parameters())+list(head.parameters()),lr=.001);target=y.clone();shuffle=torch.Generator(device='cuda').manual_seed(seed+1)
    if objective=='shuffled_labels':
     for f in range(6):
      ii=train[family[train]==f];target[ii]=y[ii[torch.randperm(len(ii),generator=shuffle,device='cuda')]]
    history=[]
    for step in range(cfg['steps']):
     out=m(X[train],membership=membership);recon=(out['reconstruction']-X[train]).square().mean();logits=head(out['code'][:,:64]).gather(1,family[train,None]).squeeze(1);label=nn.functional.binary_cross_entropy_with_logits(logits,target[train]);loss=recon+(cfg['label_weight']*label if objective!='reconstruction' else 0);opt.zero_grad();loss.backward();opt.step()
     if step%100==0:history.append(dict(step=step,reconstruction=float(recon.detach()),label=float(label.detach())))
    with torch.no_grad():
     out=m(X,membership=membership);pred=out['reconstruction'];label_pred=head(out['code'][:,:64]).gather(1,family[:,None]).squeeze(1)>0
     explicit=(out['gates']@membership>0).float();assert torch.equal(explicit,out['support']);assert torch.isfinite(pred).all()
     atom_use=(out['support'][train]>0).any(0).sum().item();active=out['support'].sum(1)
     for f in anchors+holdouts:
      for split in ['discovery','lexical','structural','joint']:
       ix=[i for i,r in enumerate(meta) if r['family']==f and r['split']==split]
       if not ix:continue
       a=torch.tensor(ix,device='cuda');pos={meta[i]['index']:i for i in ix};pairs=[r for r in rows if r['family']==f and r['split']==split];u=torch.tensor([pos[r['indices'][0]] for r in pairs],device='cuda');v=torch.tensor([pos[r['indices'][1]] for r in pairs],device='cuda');delta=X[v]-X[u];err=(pred[v]-pred[u])-delta
       records.append(dict(kind=kind,seed=seed,objective=objective,family=f,split=split,n=len(ix),native_pass=gate[f,split],whole_task_holdout=f in holdouts,reconstruction_ratio=float((pred[a]-X[a]).square().sum()/X[a].square().sum()),delta_error_ratio=float(err.square().sum()/delta.square().sum()),label_accuracy=None if f in holdouts or objective=='reconstruction' else float((label_pred[a]==y[a].bool()).float().mean()),active_atoms=float(active[a].mean())))
     audits.append(dict(kind=kind,seed=seed,objective=objective,parameters=sum(p.numel() for p in m.parameters())+sum(p.numel() for p in head.parameters()),train_used_atoms=atom_use,union_exact=True,normalized_dictionary_max_error=float((m.dictionary_matrix().norm(dim=0)-1).abs().max())))
     torch.save(dict(model=m.state_dict(),head=head.state_dict(),membership=M,center=center,scale=scale,metadata=meta,reconstruction=pred.cpu(),code=out['code'].cpu(),gates=out['gates'].cpu(),history=history),dest/f'{kind}_{objective}_{seed}.pt')
    print(kind,seed,objective,'train reconstruction',float(recon.detach()),flush=True)
 assert all(sha(p)==hashes[str(p)] for p in paths)
 result=dict(config=cfg,records=records,audits=audits,source_hashes_unchanged=True,whole_model_forwards=0,reserved_test_access=False,elapsed_seconds=time.time()-start)
 (ART/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n')
 print('COMPLETE',result['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
