"""H699: fixed-route continuous refit with aggregate/equal-output losses."""
import torch,json,time,hashlib,copy
from pathlib import Path
from feature_structures.models import ComputationDAG,binary_gate
from run_bilin18_dag_sparse_reuse import P
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_dag_fixed_route_refit'
def main():
 torch.set_num_threads(2);start=time.time();paths={'fits':ART/'bilin18_dag_sparse_reuse_captures.pt','results':ART/'bilin18_dag_sparse_reuse.json','implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};saved=torch.load(paths['fits'],weights_only=False);old=json.loads(paths['results'].read_text());torch.manual_seed(698)
 def sample(n):
  x=torch.rand(n*2,2)*4-2;return x[~((x[:,0]>0)&(x[:,1]>0))][:n]
 tx=sample(4096);vx=sample(2048);hx=torch.rand(2048,2)*2;assert torch.equal(hx,saved['heldout'])
 def target(x):u=x[:,0]*x[:,1];return torch.stack([2*u,3*u,x[:,0]],-1)
 ty,vy,hy=map(target,[tx,vx,hx]);scale=ty.square().mean();outputscale=ty.square().mean(0);mask=torch.zeros(3,6);mask[0,4]=mask[1,5]=mask[2,0]=1
 cfg=dict(experiment='H699',hashes=hashes,steps=3000,lr=.003,batch=128,design='Warm start each H698 sparse routed-readout fit. Freeze hard edges, keep node gates on. Fresh Adam continuous-weight refits using aggregate normalizedMSE or equal-output normalizedMSE. Same minibatches within seed. Independent validation selects corresponding objective every100steps including untouched incumbent. Heldout only before/after selected fit.',limitations='Warmstart optimization diagnosis, not from-scratch reliability; fixed supplied routing and learned edge support. Different loss/selection objectives; no attribution of change solely to edge freezing.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];states={}
 for seed in [41,42,43]:
  for mode in ['aggregate','equal_output']:
   m=ComputationDAG(2,3,P,products=[(),(),((0,1),),((0,1),),(),()]);m.load_state_dict(saved['states'][seed,True,.01]);edges=[binary_gate(e).detach().clone() for e in m.edge_logits]
   def forward(x,**kw):return torch.nn.functional.linear(m(x,gates=torch.ones(len(x),6),edge_gates=edges,**kw)['hidden'],m.readout.weight*mask)
   def metrics(x,y):
    with torch.no_grad():
     mse=(forward(x)-y).square().mean(0);return dict(relative_mse=float(mse.sum()/y.square().mean(0).sum()),mse_by_output=mse.tolist(),equal_output_error=float((mse/outputscale).mean()))
   initial=metrics(hx,hy);oldrec=next(r for r in old['records'] if r['seed']==seed and r['routed_readout'] and r['edge_cost']==.01);assert abs(initial['relative_mse']-oldrec['heldout_relative_mse'])<1e-6
   opt=torch.optim.Adam(m.parameters(),lr=.003);gen=torch.Generator().manual_seed(seed+69900);best=float('inf');history=[]
   for step in range(3001):
    if step%100==0:
     with torch.no_grad():
      mse=(forward(vx)-vy).square().mean(0);obj=float(mse.mean()/scale if mode=='aggregate' else (mse/outputscale).mean());history.append(dict(step=step,objective=obj))
      if obj<best:best=obj;beststep=step;beststate=copy.deepcopy(m.state_dict())
    if step==3000:break
    ii=torch.randint(len(tx),(128,),generator=gen);mse=(forward(tx[ii])-ty[ii]).square().mean(0);loss=mse.mean()/scale if mode=='aggregate' else (mse/outputscale).mean();opt.zero_grad();loss.backward();opt.step()
   m.load_state_dict(beststate)
   with torch.no_grad():
    pred=forward(hx);effect=((pred-forward(hx,remove_nodes=(2,))).square().mean(0)/hy.square().mean(0)).tolist();rec=dict(seed=seed,mode=mode,selected_step=beststep,initial=initial,final=metrics(hx,hy),node2_effect_energy_by_output=effect,edge_masks=[e.tolist() for e in edges],history=history);records.append(rec);states[seed,mode]=m.state_dict();print({k:rec[k] for k in ['seed','mode','selected_step','initial','final']},flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['final']['relative_mse'])).item() for r in records),heldout_inputs_exact_replay=True,initial_outputs_match_previous=True,whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(states=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
