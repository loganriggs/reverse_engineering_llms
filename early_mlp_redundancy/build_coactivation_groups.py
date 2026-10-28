"""M38 balanced training-coactivation grouping with frozen scalar coordinates."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def graph_objective(M,W):return float(np.sum(np.triu(W,1)*((M.T@M)>0)))
def improve(initial,W,seed,steps=20000):
 rng=np.random.default_rng(39000+seed);M=initial.copy();counts=M.T@M;upper=np.triu(W,1);value=float((upper*(counts>0)).sum());history=[dict(proposal=0,objective=value,accepted=0)];accepted=0;swaps=[]
 for step in range(steps):
  half=int(rng.integers(2));g,h=rng.choice(16,2,replace=False)+half*16;ia=int(rng.choice(np.flatnonzero(M[g])));ib=int(rng.choice(np.flatnonzero(M[h])));oldg=M[g].copy();oldh=M[h].copy();newg=oldg.copy();newh=oldh.copy();newg[ia]=0;newg[ib]=1;newh[ib]=0;newh[ia]=1
  newcounts=counts-np.outer(oldg,oldg)-np.outer(oldh,oldh)+np.outer(newg,newg)+np.outer(newh,newh);newvalue=float((upper*(newcounts>0)).sum())
  if newvalue>value+1e-12:
   M[g],M[h]=newg,newh;counts=newcounts;value=newvalue;accepted+=1;swaps.append([step,int(g),int(h),ia,ib])
  if (step+1)%2000==0:history.append(dict(proposal=step+1,objective=value,accepted=accepted))
 assert np.all(M.sum(0)==2) and np.all(M.sum(1)==8)
 return M,history,swaps
@torch.no_grad()
def main():
 torch.set_num_threads(1);pp=P/'coactivation_group_protocol.json';p=json.loads(pp.read_text());op=P/'coactivation_groups.json';assert not op.exists();mp=P/'natural_token_context.json';assert sha(mp)==p['natural_source_sha256'];m4=json.loads(mp.read_text());cp=Path(m4['cache']);assert sha(cp)==m4['cache_sha256'];data=torch.load(cp,weights_only=True);Y=data['Y'][1];tp=P/'response_training_targets.json';assert sha(tp)==p['target_source_sha256'];targetmeta=json.loads(tp.read_text());cp=Path(targetmeta['cache_path']);assert sha(cp)==targetmeta['cache_sha256'];cache=torch.load(cp,weights_only=True);schedulep=P/'joint_shared_comparison_protocol.json';assert sha(schedulep)==p['training_schedule_sha256'];schedule=json.loads(schedulep.read_text());models=[];records=[];sources=[];evidence=[];dest=A/'emlp_m38_coactivation_groups';dest.mkdir(exist_ok=True)
 for entry in p['source_models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];state=torch.load(path,weights_only=True);seed=entry['seed'];arm=entry['arm'];f=Featurizer('scalar',seed);f.load_state_dict(state['state']);mu,scale=state['mean'],state['scale'];encoder=f.encoder.double();bias=f.encoder_bias.double()
  natural=(Y[:4096].double()-mu.double())/scale.double();a=natural@encoder.T+bias;energy=a.square();cov=.5*(energy.T@energy)/len(energy);train_inputs=[]
  assignments=[x for x in schedule['target_assignments'] if x['seed']==seed];assert len(assignments)==120
  for assignment in assignments:
   c=cache['inputs'][assignment['input_key']];x=(c['write'].reshape(-1,1152).double()-mu.double())/scale.double();ac=x@encoder.T+bias;en=ac.square();cov+=.5*(en.T@en)/(120*len(en));train_inputs.append(c['write'].reshape(-1,1152))
  diag=cov.diag().sqrt();den=diag[:,None]*diag[None,:];W=torch.where(den>0,cov/den,torch.zeros_like(cov));W.fill_diagonal_(0);W=W.numpy();assert np.isfinite(W).all()
  gen=torch.Generator().manual_seed(38000+seed);M=np.zeros((32,128),dtype=np.int32)
  for half in range(2):M[np.arange(16)[:,None]+16*half,torch.randperm(128,generator=gen).numpy().reshape(16,8)]=1
  learned,history,swaps=improve(M,W,seed);source_record=dict(seed=seed,source_arm=arm,initial_objective=graph_objective(M,W),learned_objective=graph_objective(learned,W),accepted_swaps=len(swaps),history=history,zero_energy_atoms=int((diag==0).sum()));sources.append(source_record);evidence.append(dict(seed=seed,source_arm=arm,affinity=torch.from_numpy(W),random_membership=torch.from_numpy(M),learned_membership=torch.from_numpy(learned),accepted_swaps=swaps))
  cohorts=[('natural_train',Y[:4096]),('task_train',torch.cat(train_inputs)),('natural_evaluation',Y[4096:])]
  for condition,membership in [('scalar',None),('random',M),('coactivation',learned)]:
   kind='scalar' if condition=='scalar' else 'shared_fixed';model=Featurizer(kind,seed);model.load_state_dict(state['state'])
   if membership is not None:model.membership.copy_(torch.from_numpy(membership))
   for name,value in model.state_dict().items():
    if name!='membership':assert torch.equal(value,state['state'][name])
   if condition=='scalar':me=dict(entry,source_arm=arm,grouping=condition)
   else:
    fn=dest/f'{seed}_{arm}_{condition}.pt';saved=dict(state={k:v.cpu() for k,v in model.state_dict().items()},mean=mu,scale=scale,kind=kind,seed=seed,arm=arm+':'+condition,source_arm=arm,grouping=condition,source_sha256=entry['sha256'],protocol_sha256=sha(pp));torch.save(saved,fn);me=dict(kind=kind,seed=seed,arm=saved['arm'],source_arm=arm,grouping=condition,path=str(fn),sha256=sha(fn))
   models.append(me)
   for cohort,v in cohorts:
    x=(v-mu)/scale;pred,q,code=model(x);recon=pred*scale+mu;err=recon.double()-v.double();act=q.sum(-1);M2=model.membership.numpy();records.append(dict(seed=seed,source_arm=arm,grouping=condition,cohort=cohort,positions=len(v),relative_write_error=float(err.square().sum()/v.double().square().sum()),normalized_mse=float((pred.double()-x.double()).square().mean()),active_mean=float(act.double().mean()),active_min=int(act.min()),active_max=int(act.max()),used_atoms=int((code.abs().sum(0)>0).sum()),covered_atoms=128 if condition=='scalar' else int((M2.sum(0)>0).sum()),duplicate_group_pairs=0 if condition=='scalar' else sum(np.array_equal(M2[i],M2[j]) for i in range(32) for j in range(i)),graph_objective=None if condition=='scalar' else graph_objective(M2,W)))
  print('CONSTRUCTED',seed,arm,'OBJECTIVE',source_record['initial_objective'],source_record['learned_objective'],'SWAPS',len(swaps),flush=True)
 ep=A/'emlp_m38_coactivation_evidence.pt';torch.save(evidence,ep);result=dict(protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),sources=sources,models=models,records=records,evidence_path=str(ep),evidence_sha256=sha(ep),scope='Training-only coactivation graph and fixed-budget balanced swaps; no decoder training or native intervention evaluation. Local metrics are not causal fidelity or semantic-unit evidence. Task-train metric concatenates tokens whereas graph weighting is equal per scheduled endpoint.')
 op.write_text(json.dumps(result,indent=2)+'\n');print('DONE',len(models),len(records),flush=True)
if __name__=='__main__':main()
