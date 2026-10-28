"""H698: sparse edges with duplicate nonlinear routes and readout controls."""
import torch,json,time,hashlib,copy
from pathlib import Path
from feature_structures.models import ComputationDAG,binary_gate
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_dag_sparse_reuse';P=[(),(),(0,1),(0,1),(2,3),(2,3)]
def main():
 torch.set_num_threads(2);start=time.time();paths={'implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};torch.manual_seed(698)
 def sample(n):
  x=torch.rand(n*2,2)*4-2;return x[~((x[:,0]>0)&(x[:,1]>0))][:n]
 tx=sample(4096);vx=sample(2048);hx=torch.rand(2048,2)*2
 def target(x):u=x[:,0]*x[:,1];return torch.stack([2*u,3*u,x[:,0]],-1)
 ty,vy,hy=map(target,[tx,vx,hx]);scale=ty.square().mean();mask=torch.zeros(3,6);mask[0,4]=mask[1,5]=mask[2,0]=1
 cfg=dict(experiment='H698',hashes=hashes,steps=3000,lr=.003,batch=128,design='Learn continuous functions and hard-ST global edges on fixed6node graph with duplicate product nodes2/3, two consumer nodes4/5. All node gates supplied on. Compare unrestricted readout vs routed readout(4->out0,5->out1,0->out2); edgecost0or0.01,three seeds. Train/validate outsidepositivequadrant; holdpositivequadrantout. Targets(2xy,3xy,x). Select by validation normalizedMSE+edgecountcost.',limitations='Topology candidate graph supplied, not graph discovery; only edge gates learned. Routed readout imposes paths. Different effective parameter counts. Unused nodes still execute; edge sparsity is not measured runtime savings. No LLM claims.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];states={}
 for seed in [41,42,43]:
  for routed in [False,True]:
   for cost in [0.,.01]:
    torch.manual_seed(seed);m=ComputationDAG(2,3,P,products=[(),(),((0,1),),((0,1),),(),()]);opt=torch.optim.Adam(m.parameters(),lr=.003);gen=torch.Generator().manual_seed(seed+100)
    def forward(x,**kw):
     out=m(x,gates=torch.ones(len(x),6),**kw)
     return torch.nn.functional.linear(out['hidden'],m.readout.weight*mask) if routed else out['reconstruction']
    def count():return sum(binary_gate(e).sum() for e in m.edge_logits)
    best=float('inf');history=[]
    for step in range(3001):
     if step%100==0:
      with torch.no_grad():
       err=(forward(vx)-vy).square().mean()/scale;n=count();val=float(err+cost*n);history.append(dict(step=step,validation_error=float(err),edges=float(n),objective=val))
       if val<best:best=val;beststate=copy.deepcopy(m.state_dict());beststep=step
     if step==3000:break
     ii=torch.randint(len(tx),(128,),generator=gen);loss=(forward(tx[ii])-ty[ii]).square().mean()/scale+cost*count();opt.zero_grad();loss.backward();opt.step()
    m.load_state_dict(beststate)
    with torch.no_grad():
     pred=forward(hx);denom=hy.square().mean(0);interventions=[]
     for nodes in [(2,),(3,),(2,3)]:interventions.append(dict(nodes=nodes,effect_energy_by_output=((pred-forward(hx,remove_nodes=nodes)).square().mean(0)/denom).tolist()))
     edges=[]
     for child,parents in enumerate(P):
      for index,parent in enumerate(parents):edges.append(dict(parent=parent,child=child,active=int(binary_gate(m.edge_logits[child])[index]),effect_energy_by_output=((pred-forward(hx,remove_edges=((parent,child),))).square().mean(0)/denom).tolist()))
     rec=dict(seed=seed,routed_readout=routed,edge_cost=cost,selected_step=beststep,active_edges=float(count()),validation_objective=best,heldout_relative_mse=float((pred-hy).square().sum()/hy.square().sum()),heldout_mse_by_output=(pred-hy).square().mean(0).tolist(),node_interventions=interventions,edges=edges,history=history);records.append(rec);states[seed,routed,cost]=m.state_dict();print({k:rec[k] for k in ['seed','routed_readout','edge_cost','selected_step','active_edges','heldout_relative_mse','node_interventions']},flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['heldout_relative_mse'])).item() for r in records),whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(states=states,heldout=hx),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
