"""H935: discovery-only lexical CV; freeze continuous/group models for later live tests."""
from run_bilin18_binding_role_order_panel import ROOT, ART, sha
from run_bilin18_query_prototype_screen import kmeans
from pathlib import Path
import torch, numpy as np, json, time
S='bilin18_discovery_query_crossvalidation'
HEADS=[(8,3),(13,0),(14,4)]
ALPHAS=[1e-5,1e-4,1e-3,1e-2,1e-1,1.]
MODELS=[('linear',-1)]+[(kind,seed) for kind in ['kmeans','random'] for seed in range(3)]

def gate(Q, f):
 if f['kind']=='linear':return torch.zeros(len(Q),dtype=torch.long)
 if f['kind']=='kmeans':return torch.cdist(Q,f['centers']).argmin(1)
 return ((Q@f['planes']>f['thresholds']).long()*torch.tensor([1,2,4])).sum(1)

def fit_group(Q,kind,seed):
 f={'kind':kind,'seed':seed}
 if kind=='linear':return f
 if kind=='kmeans':C,lab,it=kmeans(Q,8,9350+seed);f.update(centers=C,iterations=it)
 else:
  gen=torch.Generator().manual_seed(9350+seed);P=torch.randn(256,3,generator=gen,dtype=torch.double);P=P/P.norm(dim=0);f.update(planes=P,thresholds=(Q@P).median(0).values);lab=gate(Q,f);f['centers']=torch.stack([Q[lab==c].mean(0) if (lab==c).any() else Q.mean(0) for c in range(8)])
 f['occupancy']=torch.bincount(lab,minlength=8).tolist()
 return f

def fit(Q,targets,kind,seed,alpha,group=None):
 f=dict(fit_group(Q,kind,seed) if group is None else group);labels=gate(Q,f);f['positions']={};normal=0.;dualcheck=0.
 for dist in range(max(map(len,targets))):
  ns=[n for n,y in enumerate(targets) if len(y)>dist];y=torch.stack([targets[n][-1-dist] for n in ns]);xx=Q[ns];pooled=y.mean()
  if kind=='linear':mean=xx.mean(0);base=pooled;X=xx-mean;Y=y-base;means=None
  else:
   labels_here=labels[ns];means=torch.stack([y[labels_here==c].mean() if (labels_here==c).any() else pooled for c in range(8)]);base=means[labels_here];mean=torch.zeros(256,dtype=torch.double);X=xx-f['centers'][labels_here];Y=y-base
  w=X.T@torch.linalg.solve(X@X.T+len(ns)*alpha*torch.eye(len(ns),dtype=torch.double),Y)
  normal=max(normal,float((X.T@(X@w-Y)+len(ns)*alpha*w).abs().max()))
  if dist==0:
   primal=torch.linalg.solve(X.T@X+len(ns)*alpha*torch.eye(256,dtype=torch.double),X.T@Y);dualcheck=float((w-primal).abs().max())
  f['positions'][dist]={'mean':mean,'pooled':pooled,'means':means,'weight':w,'count':len(ns)}
 f.update(alpha=alpha,normal_error=normal,primal_dual_error=dualcheck)
 return f

def predict(q,T,f,correction=True):
 lab=int(gate(q[None],f)[0]);vv=[]
 for d in reversed(range(T)):
  p=f['positions'][d]
  if f['kind']=='linear':base=p['pooled'];x=q-p['mean']
  else:base=p['means'][lab];x=q-f['centers'][lab]
  vv.append(base+(x@p['weight'] if correction else 0.))
 return torch.stack(vv)

@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6)
 paths={'runner':Path(__file__),'panel':ART/'bilin18_multidomain_matched_competence_captures.pt','queries':ART/'bilin18_query_prototype_screen_fits.pt','group_implementation':ROOT/'scripts/run_bilin18_query_prototype_screen.py'};hashes={k:sha(v) for k,v in paths.items()}
 data=torch.load(paths['panel'],weights_only=False);queries=torch.load(paths['queries'],weights_only=False)['queries'];ex=data['examples'];domains=['agreement','language','binding'];pairs={d:sorted({e['pair'] for e in ex if e['domain']==d})[:4] for d in domains};discovery=[i for i,e in enumerate(ex) if e['domain'] in domains and e['pair'] in pairs[e['domain']]];halves=[[i for i in discovery if ex[i]['pair'] in pairs[ex[i]['domain']][2*f:2*f+2]] for f in range(2)];assert len(discovery)==96 and all(len(a)==48 for a in halves) and not set(halves[0])&set(halves[1])
 cfg={'experiment':'H935','hashes':hashes,'alphas':ALPHAS,'models':MODELS,'heads':HEADS,'discovery_indices':discovery,'fold_halves':halves,'selection':'Per head and model/seed minimize summed held-out persistent-write squared error over both lexical folds. Ties choose stronger regularization. Refit all96 discovery, no transfer scoring. Both formats retained in each fold.','fit':'Mean coefficient squared loss plus alpha norm2; linear unpenalized intercept, grouped mean plus shared continuous residual. Q=q/q2 concatenated divided16. Groups refit within fold.','random_control':'Three seeded random hyperplanes, train median thresholds, eight bit-coded groups. Empty groups use global query center and global distance coefficient. Same eight-center/shared-residual form; occupancy and partition complexity not exactly matched. Random geometric groups may contain meaningful context.','limitations':'Selection uses cached native queries and local writes, not downstream KL or donor effects. This is not a causal validation or untouched test. Existing research-transfer data loaded in shared cache but never indexed for fitting or scoring.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 records=[];fits={};occupancy=[];accessed=set();normal=0.;dual=0.
 for l,h in HEADS:
  for kind,seed in MODELS:
   for fold in range(2):
    tr=halves[fold];va=halves[1-fold];accessed.update(tr+va);Q=torch.stack([queries[i,l,h] for i in tr])/16;targets=[data['captures'][i][l,h]['routing'].double() for i in tr];group=fit_group(Q,kind,seed)
    occupancy.append(dict(layer=l,head=h,kind=kind,seed=seed,fold=fold,counts=group.get('occupancy',[])))
    for alpha in ALPHAS:
     f=fit(Q,targets,kind,seed,alpha,group);normal=max(normal,f['normal_error']);dual=max(dual,f['primal_dual_error'])
     for i in va:
      A=data['captures'][i][l,h]['routing'].double();V=data['bus'][i][:,h].double();q=queries[i,l,h]/16;pred=predict(q,len(A),f);base=predict(q,len(A),f,False);err=(pred-A)@V;be=(base-A)@V;ref=A@V
      records.append(dict(layer=l,head=h,kind=kind,seed=seed,fold=fold,alpha=alpha,index=i,domain=ex[i]['domain'],family=ex[i]['family'],write_error_sq=float(err.square().sum()),baseline_error_sq=float(be.square().sum()),reference_sq=float(ref.square().sum()),routing_error_sq=float((pred-A).square().sum())))
   print('CV',l,h,kind,seed,flush=True)
 selected=[];grid=[]
 for l,h in HEADS:
  Q=torch.stack([queries[i,l,h] for i in discovery])/16;targets=[data['captures'][i][l,h]['routing'].double() for i in discovery]
  for kind,seed in MODELS:
   rows=[r for r in records if (r['layer'],r['head'],r['kind'],r['seed'])==(l,h,kind,seed)];ss=[]
   for alpha in ALPHAS:
    rr=[r for r in rows if r['alpha']==alpha];assert len(rr)==96 and {r['index'] for r in rr}==set(discovery);err=sum(r['write_error_sq'] for r in rr);ref=sum(r['reference_sq'] for r in rr);base=sum(r['baseline_error_sq'] for r in rr);s=dict(layer=l,head=h,kind=kind,seed=seed,alpha=alpha,write_error_sq=err,write_error_ratio=(err/ref)**.5,baseline_error_ratio=(base/ref)**.5);ss.append(s);grid.append(s)
   chosen=min(ss,key=lambda s:(s['write_error_sq'],-s['alpha']));selected.append(chosen);f=fit(Q,targets,kind,seed,chosen['alpha']);f['train_indices']=discovery;fits[l,h,kind,seed]=f;normal=max(normal,f['normal_error']);dual=max(dual,f['primal_dual_error'])
 audits={'hashes_unchanged':all(sha(v)==hashes[k] for k,v in paths.items()),'folds_disjoint':not bool(set(halves[0])&set(halves[1])),'accessed_indices':sorted(accessed),'only_discovery_indexed':accessed==set(discovery),'normal_equation_max_abs':normal,'primal_dual_max_abs':dual,'finite':all(np.isfinite(r['write_error_sq']) for r in records),'research_transfer_evaluations':0,'reserved_test_evaluations':0,'whole_model_forwards':0,'frozen_models':len(fits)}
 assert audits['hashes_unchanged'] and audits['only_discovery_indexed'] and audits['finite'] and normal<1e-9 and dual<1e-8,audits
 torch.save({'fits':fits,'selected':selected,'discovery_indices':discovery,'config':cfg},ART/f'{S}_fits.pt')
 audits['frozen_fits_sha256']=sha(ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,grid=grid,selected=selected,occupancy=occupancy,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(json.dumps(dict(selected=selected,audits=audits),indent=2),flush=True)
if __name__=='__main__':main()
