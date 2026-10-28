"""H695: exact expected reconstruction cost under independent Bernoulli gates."""
import torch,json,time,hashlib,copy
from pathlib import Path
from run_bilin18_shared_gate_discovery import EnergyGates
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_shared_gate_expected_cost'
def pattern_probabilities(logits,patterns):
 p=logits.sigmoid()[:,None,:]
 return (p*patterns[None,:,:]+(1-p)*(1-patterns[None,:,:])).prod(-1)
def expected_cost(logits,patterns,costs):
 return (pattern_probabilities(logits,patterns)*costs).sum(-1).mean()
def verify_expected_cost():
 torch.manual_seed(695);patterns=torch.tensor(list(__import__('itertools').product([0.,1.],repeat=4)),dtype=torch.float64);logits=torch.randn(3,4,dtype=torch.float64,requires_grad=True);costs=torch.rand(3,16,dtype=torch.float64)
 probs=pattern_probabilities(logits,patterns);assert torch.allclose(probs.sum(-1),torch.ones(3,dtype=torch.float64),atol=1e-12)
 assert torch.autograd.gradcheck(lambda z:expected_cost(z,patterns,costs),(logits,),eps=1e-6,atol=1e-6)
 actual=torch.autograd.grad(expected_cost(logits,patterns,costs),logits)[0];analytic=torch.zeros_like(actual)
 for g in range(4):
  other=[i for i in range(4) if i!=g];p=logits.sigmoid();otherprob=(p[:,None,other]*patterns[None,:,other]+(1-p[:,None,other])*(1-patterns[None,:,other])).prod(-1)
  difference=(otherprob*costs*(2*patterns[:,g]-1)[None,:]).sum(-1)
  analytic[:,g]=p[:,g]*(1-p[:,g])*difference/len(logits)
 error=float((actual-analytic).abs().max().detach());assert error<1e-12
 return dict(probability_normalization=True,finite_difference_gradcheck=True,analytic_gradient_max_error=error)
def main():
 torch.set_num_threads(2);verification=verify_expected_cost();start=time.time();paths={'data':ART/'bilin18_shared_exact_gates_captures.pt','baseline':ART/'bilin18_shared_gate_discovery.json','gate_implementation':ROOT/'scripts/run_bilin18_shared_gate_discovery.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);tables=data['tables'];patterns=data['patterns'];scale=data['scale'];cost=.01;weights=torch.tensor([8,4,2,1]);baseline=json.loads(paths['baseline'].read_text())
 tx=tables['train']['x'];trainobj=tables['train']['normalized_errors']+cost*patterns.sum(-1);
 cfg=dict(experiment='H695',hashes=hashes,steps=3000,lr=.003,batch=128,group_cost=cost,design='Same H692 architecture, initialization seeds and batch streams. Train exact expected reconstruction-plus-count cost over all16patterns under independent Bernoulli probabilities; original0.01cost, no teacher labels or straight-through gradient. Select checkpoint every100steps by validation reconstruction-plus-count objective, as H692. Evaluate held combinations only after selection.',limitations='Frozen representation originally learned with supplied gates. Not joint unsupervised learning. Stochastic expected objective differs from hard-threshold evaluation; factorized gate distribution. Extra exact-search compute. Only three seeds and one teacher.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];states={}
 def evaluate(net,split):
  t=tables[split];x,g,errors=t['x'],t['g'],t['normalized_errors']
  with torch.no_grad():
   pred=(net(x)>0).float();idx=(pred*weights).sum(-1).long();rows=torch.arange(len(x));objective=errors+cost*patterns.sum(-1);opt=objective.argmin(-1);chosenerr=errors[rows,idx];chosenobj=objective[rows,idx];return dict(relative_mse=float(chosenerr.mean()*scale/x.square().mean()),objective=float(chosenobj.mean()),objective_gap=float((chosenobj-objective[rows,opt]).mean()),exact_search_agreement=float((idx==opt).float().mean()),teacher_group_recovery=float((pred==g).all(-1).float().mean()),mean_groups=float(pred.sum(-1).mean()),expected_stochastic_objective=float(expected_cost(net(x),patterns,objective)))
 for seed in [21,22,23]:
  for kind in ['linear','energy']:
   torch.manual_seed(seed);net=torch.nn.Linear(16,4) if kind=='linear' else EnergyGates();opt=torch.optim.Adam(net.parameters(),lr=.003);gen=torch.Generator().manual_seed(seed+100);best=float('inf');history=[]
   for step in range(3001):
    if step%100==0:
     val=evaluate(net,'validation');history.append(dict(step=step,**val))
     if val['objective']<best:best=val['objective'];beststate=copy.deepcopy(net.state_dict());beststep=step
    if step==3000:break
    ii=torch.randint(len(tx),(128,),generator=gen);loss=expected_cost(net(tx[ii]),patterns,trainobj[ii]);opt.zero_grad();loss.backward();opt.step()
   net.load_state_dict(beststate);rec=dict(seed=seed,kind=kind,selected_step=beststep,train=evaluate(net,'train'),validation=evaluate(net,'validation'),heldout=evaluate(net,'heldout'),history=history);records.append(rec);states[kind,seed]=beststate;print({k:rec[k] for k in ['seed','kind','selected_step','validation','heldout']},flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['heldout']['objective'])).item() for r in records),expected_cost_verification=verification,whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(states=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
