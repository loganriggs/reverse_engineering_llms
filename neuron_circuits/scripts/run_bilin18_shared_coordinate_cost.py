"""H701: joint label-free learning with a factorial oracle-information audit."""
import torch,json,time,hashlib,copy,itertools
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from feature_structures.models import SharedGroups,binary_gate,union_support
from run_bilin18_shared_gate_discovery import EnergyGates
from run_bilin18_shared_gate_expected_cost import expected_cost
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_shared_coordinate_cost'
def main():
 torch.set_num_threads(2);start=time.time();paths={'data':ART/'bilin18_shared_exact_gates_captures.pt','teacher':ART/'feature_structure_synthetic_captures.pt','implementation':ROOT/'scripts/feature_structures/models.py','gate_implementation':ROOT/'scripts/run_bilin18_shared_gate_discovery.py','cost_implementation':ROOT/'scripts/run_bilin18_shared_gate_expected_cost.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);tables=data['tables'];patterns=data['patterns'];scale=data['scale'];teacher=torch.load(paths['teacher'],weights_only=False);B,M=teacher['true_dictionary'],teacher['true_membership'];cost=.01;weights=torch.tensor([8,4,2,1]);tx=tables['train']['x'];tg=tables['train']['g'];records=[];states={};max_reference_error=0.
 cfg=dict(experiment='H701',hashes=hashes,steps=3000,lr=.003,batch=128,group_cost=cost,coordinate_costs=[.005,.02,.05],design='Fresh random dictionary, coordinate encoder, membership and energy detector. Joint label-free only. Add active-support coordinate costs0.005/0.02/0.05 to groupcost0.01; same initializations/batches as H700 joint baselines. For unknown gates train exact Bernoulli expected cost over16patterns; unknown membership uses hardTop3 ST. Same three seeds/minibatches. Select hard validation reconstruction+count cost every100steps, held combinations afterward. Joint condition never uses generating labels in optimization/selection.',limitations='One noiseless teacher,known latent/group sizes and3members pergroup,small exactenumeration,hardmembership surrogate. Different cost arms are tradeoffs, not selected on heldout. Counts charge active support; with continuous coordinates nonzero almost surely this equals codeL0 except exact zeros. Fixed membership costs constant; no learned membershipbudget.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2))
 for seed in [51,52,53]:
  for coordcost in [.005,.02,.05]:
   mode='joint'
   torch.manual_seed(seed);m=SharedGroups(16,8,4,None,3);net=EnergyGates();knownM=mode in ['oracle_both','oracle_membership'];knownG=mode in ['oracle_both','oracle_gates'];opt=torch.optim.Adam(list(m.parameters())+list(net.parameters()),lr=.003);gen=torch.Generator().manual_seed(seed+100);best=float('inf');history=[]
   def membership():return M if knownM else binary_gate(m.membership_logits,3)
   def error_table(x):
    q=union_support(patterns,membership());a=m.coordinate_encoder(x);pred=torch.einsum('npa,da->npd',a[:,None,:]*q[None,:,:],m.dictionary_matrix())+m.bias;return (pred-x[:,None,:]).square().mean(-1)/scale
   def evaluation(split):
    x,g=tables[split]['x'],tables[split]['g']
    with torch.no_grad():
     errors=error_table(x);s=g if knownG else (net(x)>0).float();idx=(s*weights).sum(-1).long();rows=torch.arange(len(x));obj=errors+cost*patterns.sum(-1)+coordcost*union_support(patterns,membership()).sum(-1);bestidx=obj.argmin(-1);return dict(relative_mse=float(errors[rows,idx].mean()*scale/x.square().mean()),objective=float(obj[rows,idx].mean()),exact_inference_objective=float(obj[rows,bestidx].mean()),exact_inference_relative_mse=float(errors[rows,bestidx].mean()*scale/x.square().mean()),mean_groups=float(s.sum(-1).mean()),mean_active_atoms=float(union_support(s,membership()).sum(-1).mean()),unique_gate_patterns=int(torch.unique(s,dim=0).shape[0]),largest_pattern_fraction=float(torch.unique(s,dim=0,return_counts=True)[1].max()/len(s)),gate_counts=s.sum(0).tolist())
   with torch.no_grad():
    errors=error_table(tx[:32]);direct=(m(tx[:32],gates=patterns[5].expand(32,-1),membership=membership())['reconstruction']-tx[:32]).square().mean(-1)/scale;diff=float((errors[:,5]-direct).abs().max());max_reference_error=max(max_reference_error,diff);assert diff<1e-5
   for step in range(3001):
    if step%100==0:
     val=evaluation('validation');history.append(dict(step=step,**val))
     if val['objective']<best:best=val['objective'];beststep=step;beststate=(copy.deepcopy(m.state_dict()),copy.deepcopy(net.state_dict()))
    if step==3000:break
    ii=torch.randint(len(tx),(128,),generator=gen);errors=error_table(tx[ii]);obj=errors+cost*patterns.sum(-1)+coordcost*union_support(patterns,membership()).sum(-1)
    loss=obj[torch.arange(len(ii)),(tg[ii]*weights).sum(-1).long()].mean() if knownG else expected_cost(net(tx[ii]),patterns,obj)
    opt.zero_grad();loss.backward();opt.step()
   m.load_state_dict(beststate[0]);net.load_state_dict(beststate[1])
   with torch.no_grad():
    L=membership();score=(B.T@m.dictionary_matrix()).abs();aa,bb=linear_sum_assignment(-score.numpy());perms=list(itertools.permutations(range(4))) if mode=='joint' else [tuple(range(4))];perm=max(perms,key=lambda p:float((L[list(p)][:,bb]==M).float().mean()));agreement=float((L[list(perm)][:,bb]==M).float().mean());x,g=tables['heldout']['x'],tables['heldout']['g'];s=g if knownG else (net(x)>0).float();rec=dict(seed=seed,mode=mode,coordinate_cost=coordcost,selected_step=beststep,train=evaluation('train'),validation=evaluation('validation'),heldout=evaluation('heldout'),membership=L.tolist(),matched_atom_cosine=float(score[aa,bb].mean()),posthoc_membership_entry_agreement=agreement,posthoc_group_permutation=perm,heldout_gate_exact_agreement=float((s[:,list(perm)]==g).all(-1).float().mean()),history=history);records.append(rec);states[seed,coordcost]=beststate;print({k:rec[k] for k in ['seed','coordinate_cost','selected_step','heldout','matched_atom_cosine','posthoc_membership_entry_agreement','heldout_gate_exact_agreement']},flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['heldout']['relative_mse'])).item() for r in records),max_reference_forward_error=max_reference_error,whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(states=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
