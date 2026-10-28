"""H693: exact conditional gate inference versus learned detectors."""
import torch,json,time,hashlib,itertools
from pathlib import Path
from feature_structures.models import SharedGroups,binary_gate
from run_bilin18_shared_gate_discovery import EnergyGates
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_shared_exact_gates'
def main():
 torch.set_num_threads(2);start=time.time();paths={'teacher':ART/'feature_structure_synthetic_captures.pt','fit':ART/'bilin18_shared_discrete_search_captures.pt','detectors':ART/'bilin18_shared_gate_discovery_captures.pt','previous_results':ART/'bilin18_shared_gate_discovery.json','implementation':ROOT/'scripts/feature_structures/models.py','gate_implementation':ROOT/'scripts/run_bilin18_shared_gate_discovery.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
 teacher=torch.load(paths['teacher'],weights_only=False);B,M=teacher['true_dictionary'],teacher['true_membership'];fit=torch.load(paths['fit'],weights_only=False)['states'][11];detectors=torch.load(paths['detectors'],weights_only=False)['states'];old=json.loads(paths['previous_results'].read_text());model=SharedGroups(16,8,4,None,3)
 with torch.no_grad():
  for target,p in zip([model.dictionary,model.coordinate_encoder.weight,model.coordinate_encoder.bias,model.bias],fit['parameters']):target.copy_(p[0])
  model.membership_logits.copy_(fit['membership'][0]*4-2)
 model.eval();trainsets=[s for n in [1,2] for s in itertools.combinations(range(4),n)];heldsets=[s for n in [3,4] for s in itertools.combinations(range(4),n)];gen=torch.Generator().manual_seed(692)
 def sample(sets,n):
  g=torch.zeros(n,4);which=torch.randint(len(sets),(n,),generator=gen)
  for i,k in enumerate(which):g[i,list(sets[int(k)])]=1
  a=torch.randn(n,8,generator=gen);return (a*(g@M>0))@B.T,g
 data={'train':sample(trainsets,4096),'validation':sample(trainsets,2048),'heldout':sample(heldsets,2048)};scale=data['train'][0].square().mean();patterns=torch.tensor(list(itertools.product([0.,1.],repeat=4)));weights=torch.tensor([8,4,2,1]);costs=[0.,.00001,.001,.01,.1]
 cfg=dict(experiment='H693',hashes=hashes,group_costs=costs,design='Enumerate16binary patterns per input with frozen H691 seed11 representation and identical H692 data. Compare exact per-example minima with H692 learned gates at cost0.01. Other costs are diagnostic sweeps, not heldout tuning. Teacher gate labels only evaluated after selecting by reconstruction plus group count.',limitations='Exact conditional inference only; fixed dictionary learned with teacher gates; enumeration does not scale to many groups; sweep exploratory.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];comparisons=[];tables={};max_replay=0.;max_direct=0.
 with torch.no_grad():
  for split,(x,g) in data.items():
   errors=torch.stack([(model(x,gates=p.expand(len(x),-1))['reconstruction']-x).square().mean(-1)/scale for p in patterns],dim=1);tables[split]=dict(x=x,g=g,normalized_errors=errors)
   trueidx=(g*weights).sum(-1).long();rows=torch.arange(len(x));teachererr=errors[rows,trueidx];direct=(model(x,gates=g)['reconstruction']-x).square().mean(-1)/scale;max_direct=max(max_direct,float((teachererr-direct).abs().max()))
   for cost in costs:
    objective=errors+cost*patterns.sum(-1);best=objective.argmin(-1);chosen=patterns[best];err=errors[rows,best];rec=dict(split=split,cost=cost,relative_mse=float(err.mean()*scale/x.square().mean()),objective=float(objective[rows,best].mean()),mean_groups=float(chosen.sum(-1).mean()),exact_group_recovery=float((chosen==g).all(-1).float().mean()),false_positive_groups=float(((chosen>g).float().sum(-1)).mean()),false_negative_groups=float(((chosen<g).float().sum(-1)).mean()),teacher_objective=float((teachererr+cost*g.sum(-1)).mean()));records.append(rec)
    assert (objective[rows,best]<=objective[rows,trueidx]+1e-7).all()
    if cost==.01:exact_obj=objective[rows,best];exact_best=best;print(rec,flush=True)
   for (kind,seed),state in detectors.items():
    net=torch.nn.Linear(16,4) if kind=='linear' else EnergyGates();net.load_state_dict(state);s=binary_gate(net(x));idx=(s*weights).sum(-1).long();obj=errors[rows,idx]+.01*s.sum(-1);gap=obj-exact_obj;assert gap.min()>-1e-7
    mse=float(errors[rows,idx].mean()*scale/x.square().mean());expected=next(r for r in old['records'] if r['seed']==seed and r['kind']==kind)[split]['relative_mse'];max_replay=max(max_replay,abs(mse-expected));assert abs(mse-expected)<1e-6
    comparisons.append(dict(split=split,kind=kind,seed=seed,objective=float(obj.mean()),mean_objective_gap=float(gap.mean()),exact_optimum_agreement=float((idx==exact_best).float().mean())))
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['objective'])).item() for r in records),max_previous_metric_replay_error=max_replay,max_direct_teacher_error=max_direct,all_detector_objectives_above_exact=True,whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(tables=tables,patterns=patterns,scale=scale),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,comparisons=comparisons,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
