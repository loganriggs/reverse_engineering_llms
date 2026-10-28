"""H930 task-blind pre-rotary query prototypes, routing and write prediction."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict,Counter
import torch,numpy as np,json,time
S='bilin18_query_prototype_screen';HEADS=[(8,3),(13,0),(14,4)]
def kmeans(X,k,seed):
 gen=torch.Generator().manual_seed(seed);centers=[X[int(torch.randint(len(X),(1,),generator=gen))]]
 for _ in range(1,k):
  d=torch.cdist(X,torch.stack(centers)).square().min(1).values;centers.append(X[int(torch.multinomial(d/d.sum(),1,generator=gen))])
 C=torch.stack(centers)
 for it in range(200):
  ds=torch.cdist(X,C).square();labels=ds.argmin(1);new=[]
  for c in range(k):new.append(X[labels==c].mean(0) if (labels==c).any() else X[ds.min(1).values.argmax()])
  new=torch.stack(new)
  if torch.equal(torch.cdist(X,new).argmin(1),labels):C=new;break
  C=new
 labels=torch.cdist(X,C).argmin(1);assert all((labels==c).any() for c in range(k));assert torch.isfinite(C).all()
 return C,labels,it+1

def ari(x,y):
 n=len(x);choose=lambda v:v*(v-1)/2;cells=Counter(zip(x,y));aa=Counter(x);bb=Counter(y);a=sum(choose(v) for v in aa.values());b=sum(choose(v) for v in bb.values());v=sum(choose(v) for v in cells.values());expected=a*b/choose(n);den=(a+b)/2-expected;return float((v-expected)/den) if den else 1.
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_multidomain_matched_competence_captures.pt','prior_fits':ART/'bilin18_multidomain_coefficient_prediction_fits.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H930','hashes':hashes,'heads':HEADS,'counts':[1,2,4,8],'seeds':[0,1,2],'design':'Perhead kmeans of concatenated normalized pre-rotary final q/q2, without labels or outcomes. First4 lexicalpairs/domain train,last4 evaluate. Prototype-conditioned end-relative coefficient means; uncovered distances fall back to discovery pooled-distance mean. Compare local routing and persistent-write errors, not causal fidelity. Native keys/values not used in clustering; coefficient targets are discovery routing. Context labels used only retrospective reporting/baseline.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];allix=[i for i,e in enumerate(ex) if e['domain'] in ['agreement','language','binding']];pairs={d:sorted({ex[i]['pair'] for i in allix if ex[i]['domain']==d}) for d in ['agreement','language','binding']};train=[i for i in allix if ex[i]['pair'] in pairs[ex[i]['domain']][:4]];test=[i for i in allix if i not in train];prior=torch.load(paths['prior_fits'],weights_only=False)['fits'];model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 stash={};handles=[];originals={};captures={};replay=0.;routeerr=0.;averr=0.;actual=0
 for l,h in HEADS:
  att=model.blocks[l].attn
  for which in ['q','q2']:
   def cap(mod,args,out,l=l,h=h,which=which):
    z=out[:,-1].reshape(-1,9,128)[:,h];stash[l,h,which]=torch.nn.functional.rms_norm(z,(128,)).detach().cpu()
   handles.append(getattr(att,'c_'+which).register_forward_hook(cap))
  fn=att.squared_attention;originals[l]=fn
  def attention(q,k,v,q2,k2,l=l,h=h,fn=fn):
   A=((torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128));y=fn(q,k,v,q2,k2);stash[l,h,'route']=A[:,h].detach().cpu();stash[l,h,'av_error']=float(((A[:,h,:,None]*v[:,:,h]).sum(1)-y[:,h,-1]).abs().max());return y
  att.squared_attention=attention
 buckets=defaultdict(list)
 for i in allix:buckets[len(ex[i]['ids'])].append(i)
 try:
  for length,ii in buckets.items():
   for st in range(0,len(ii),8):
    batch=ii[st:st+8];y=final_logits(model,torch.tensor([ex[i]['ids'] for i in batch],device='cuda')).double().cpu();actual+=len(batch)
    for n,i in enumerate(batch):
     replay=max(replay,float((y[n]-data['native'][i]).abs().max()))
     for l,h in HEADS:
      captures[i,l,h]=torch.cat([stash[l,h,'q'][n],stash[l,h,'q2'][n]]).double();routeerr=max(routeerr,float((stash[l,h,'route'][n]-data['captures'][i][l,h]['routing']).abs().max()));averr=max(averr,stash[l,h,'av_error'])
 finally:
  for handle in handles:handle.remove()
  for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
 fits={};records=[];assignments={};composition=[]
 for l,h in HEADS:
  X=torch.stack([captures[i,l,h] for i in train]);Y=torch.stack([captures[i,l,h] for i in test])
  for k in [1,2,4,8]:
   for seed in [0,1,2]:
    C,labels,it=kmeans(X,k,9300+seed);testlabels=torch.cdist(Y,C).argmin(1);assignments[l,h,k,seed]=testlabels.tolist();bins=defaultdict(list)
    for n,i in enumerate(train):
     A=data['captures'][i][l,h]['routing']
     for p,a in enumerate(A):bins[int(labels[n]),len(A)-1-p].append(float(a))
    means={key:float(np.mean(v)) for key,v in bins.items()};fits[l,h,k,seed]={'centers':C,'coefficients':means,'iterations':it,'train_indices':train,'train_labels':labels.tolist()}
    for cluster in range(k):composition.append(dict(layer=l,head=h,k=k,seed=seed,cluster=cluster,discovery_counts=dict(Counter(ex[i]['domain']+'/'+ex[i]['family']+'/'+ex[i]['format'] for n,i in enumerate(train) if int(labels[n])==cluster))))
    for n,i in enumerate(test):
     e=ex[i];A=data['captures'][i][l,h]['routing'].double();T=len(A);missing=0;pv=[]
     for p in range(T):
      key=int(testlabels[n]),T-1-p
      if key not in means:missing+=1
      pv.append(means.get(key,prior['distance_mean',l,h,T-1-p]))
     pred=torch.tensor(pv,dtype=torch.double);V=data['bus'][i][:,h].double();z=(A[:,None]*V).sum(0);err=((pred-A)[:,None]*V).sum(0);records.append(dict(index=i,domain=e['domain'],family=e['family'],format=e['format'],layer=l,head=h,k=k,seed=seed,route_error_sq=float((pred-A).square().sum()),route_reference_sq=float(A.square().sum()),write_error_sq=float(err.square().sum()),write_reference_sq=float(z.square().sum()),fallback_positions=missing,positions=T))
 summary=[]
 for l,h in HEADS:
  for k in [1,2,4,8]:
   for seed in [0,1,2]:
    rr=[r for r in records if (r['layer'],r['head'],r['k'],r['seed'])==(l,h,k,seed)];ratio=lambda a,b:float((sum(r[a] for r in rr)/sum(r[b] for r in rr))**.5);summary.append(dict(layer=l,head=h,k=k,seed=seed,route_error_ratio=ratio('route_error_sq','route_reference_sq'),write_error_ratio=ratio('write_error_sq','write_reference_sq'),fallback_fraction=sum(r['fallback_positions'] for r in rr)/sum(r['positions'] for r in rr)))
 stability=[dict(layer=l,head=h,k=k,seeds=[a,b],ARI=ari(assignments[l,h,k,a],assignments[l,h,k,b])) for l,h in HEADS for k in [1,2,4,8] for a,b in [(0,1),(0,2),(1,2)]]
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_abs':replay,'routing_replay_max_abs':routeerr,'AV_max_abs':averr,'fit_eval_disjoint':not bool(set(train)&set(test)),'whole_model_forwards':actual,'expected_forwards':192,'finite':all(np.isfinite(r['write_error_ratio']) for r in summary),'test_evaluations':0}
 assert audits['hashes_unchanged'] and replay<1e-4 and routeerr<1e-4 and averr<1e-3 and actual==192 and audits['finite'] and audits['fit_eval_disjoint'],audits
 torch.save({'queries':captures,'fits':fits,'assignments':assignments},ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'stability':stability,'composition':composition,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(summary,stability,audits,flush=True)
if __name__=='__main__':main()
