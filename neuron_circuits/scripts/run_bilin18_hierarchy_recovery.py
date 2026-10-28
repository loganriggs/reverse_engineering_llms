"""H696: trained nested union and additive refinements, causal factor recovery."""
import torch,json,time,hashlib,itertools,copy
from pathlib import Path
from feature_structures.models import AdditiveRefinements,implication_closure,union_support
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_hierarchy_recovery';PARENTS=[(),(0,),(0,),(1,),(1,),(2,),(2,)]
class HierarchyFit(torch.nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.decoder=AdditiveRefinements(20,2,PARENTS);self.encoder=torch.nn.Linear(20,14);self.bias=torch.nn.Parameter(torch.zeros(20));self.register_buffer('nested',implication_closure(torch.eye(7),PARENTS).repeat_interleave(2,-1))
 def forward(self,x,seeds,remove=None):
  z=self.encoder(x).reshape(-1,7,2)
  if remove is not None:
   keep=torch.ones(7,device=x.device);keep[remove]=0;z=z*keep[None,:,None]
  if self.kind=='additive':return self.decoder(z,seeds)[0]+self.bias
  if self.kind=='taxonomy':q=union_support(implication_closure(seeds,PARENTS),self.nested)
  else:q=seeds.repeat_interleave(2,-1) # flat caller supplies expanded gates
  D=torch.nn.functional.normalize(self.decoder.dictionary,dim=1).permute(1,0,2).reshape(20,14)
  return (z.reshape(-1,14)*q)@D.T+self.bias

def main():
 torch.set_num_threads(2);start=time.time();paths={'implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};torch.manual_seed(696);B=torch.linalg.qr(torch.randn(20,14)).Q
 trainsets=[(i,) for i in range(7)]+list(itertools.combinations([3,4,5,6],2));heldsets=list(itertools.combinations([3,4,5,6],3))+[(3,4,5,6)]
 def sample(sets,n):
  s=torch.zeros(n,7);which=torch.randint(len(sets),(n,))
  for i,k in enumerate(which):s[i,list(sets[int(k)])]=1
  g=implication_closure(s,PARENTS);a=torch.randn(n,14);code=a*g.repeat_interleave(2,-1);return code@B.T,s,g,code
 tx,ts,tg,tc=sample(trainsets,4096);hx,hs,hg,hc=sample(heldsets,2048);cfg=dict(experiment='H696',hashes=hashes,steps=3000,lr=.003,batch=128,design='Seven-node tree,2teacher directions per node, orthonormal14atom dictionary in20dimensions. Train single-node seeds and all leaf pairs, hold out leaf triples/all4. Supplied tree, seeds and taxonomy memberships. Train nested-union/additive-refinement/flat expanded-gate models from identical weights and minibatches. Matched594trainable parameters. Measure functional transfer, block subspace recovery and learned-component removal against true refinements.',limitations='Known hierarchy and gates; flat control receives closure as side information. Models intentionally functionally equivalent with this grouping, so no hierarchy superiority claim. One teacher,three seeds. Coordinates and dictionary are learned, topology is not.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];states={};max_initial_difference=0.
 for seed in [31,32,33]:
  torch.manual_seed(seed);base=HierarchyFit('taxonomy').state_dict();reference=None
  for kind in ['taxonomy','additive','flat']:
   m=HierarchyFit(kind);m.load_state_dict(base);gtrain=tg if kind=='flat' else ts;gheld=hg if kind=='flat' else hs
   with torch.no_grad():
    initial=m(tx[:64],gtrain[:64]);reference=initial if reference is None else reference;max_initial_difference=max(max_initial_difference,float((initial-reference).abs().max()))
   opt=torch.optim.Adam(m.parameters(),lr=.003);gen=torch.Generator().manual_seed(seed+100)
   for step in range(3000):
    ii=torch.randint(len(tx),(128,),generator=gen);loss=(m(tx[ii],gtrain[ii])-tx[ii]).square().mean();opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():
    pred=m(hx,gheld);subspaces=[];interventions=[]
    for node in range(7):
     Q=torch.linalg.qr(m.decoder.dictionary[node]).Q;teacher=B[:,2*node:2*node+2];subspaces.append(float((Q.T@teacher).square().sum()/2));removed=pred-m(hx,gheld,remove=node);truth=hc[:,2*node:2*node+2]@teacher.T;active=hg[:,node]>0
     interventions.append(dict(node=node,relative_effect_error=float((removed[active]-truth[active]).square().sum()/truth[active].square().sum()),off_target_energy_fraction=float((removed[active]-removed[active]@teacher@teacher.T).square().sum()/removed[active].square().sum())))
    rec=dict(seed=seed,kind=kind,parameters=sum(p.numel() for p in m.parameters()),train_relative_mse=float((m(tx,gtrain)-tx).square().sum()/tx.square().sum()),heldout_relative_mse=float((pred-hx).square().sum()/hx.square().sum()),block_subspace_overlap=subspaces,interventions=interventions);records.append(rec);states[kind,seed]=m.state_dict();print(rec,flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['heldout_relative_mse'])).item() for r in records),initial_forward_max_difference=max_initial_difference,whole_model_forwards=0);assert max_initial_difference<1e-5 and all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(states=states,teacher=B,heldout=(hx,hs,hg,hc)),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
