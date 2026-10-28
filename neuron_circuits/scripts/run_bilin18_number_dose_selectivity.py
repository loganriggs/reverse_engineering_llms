"""H889 frozen number direction dose, norm, layer, and copying controls."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
import torch,numpy as np,json,time
from pathlib import Path
from collections import defaultdict
S='bilin18_number_dose_selectivity';SETS={'core':list(range(11,17)),'complement':[8,9,10,17]}|{f'L{k}':[k] for k in range(11,17)}
CASES=[('core','identity',0.,0)]+[('core',m,d,0) for m in ['raw','sphere'] for d in [.25,.5,1.]]+[('core','radial',1.,0)]+[('core','random',1.,s) for s in range(4)]+[(g,m,1.,0) for g in ['complement']+[f'L{k}' for k in range(11,17)] for m in ['raw','sphere']]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'panel':ART/'bilin18_number_transfer_screen_captures.pt','selection':ART/'bilin18_number_transfer_screen_analysis.json','fits':ART/'bilin18_agreement_partner_number_fits.pt','centroids':ART/'bilin18_agreement_partner_number.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H889','hashes':hashes,'sets':SETS,'cases':CASES,'design':'Both frozen task axes on120 eligible task+copy contexts. Raw opposite-number clamps dose.25/.5/1; exact norm-and-coordinate-preserving sphere edit; norm-only radial control; four equal-angle random sphere controls. Core, complement and each core layer. Fixed conditions, no refit, no reserved test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];eligible=json.loads(paths['selection'].read_text())['paired_competent_cells'];selection=[i for i,e in enumerate(ex) if [e['cohort'],e['format']] in eligible];assert len(selection)==120;fits=torch.load(paths['fits'],weights_only=False);old=json.loads(paths['centroids'].read_text());cc={(r['format'],r['layer']):torch.tensor(r['centroids'],device='cuda',dtype=torch.double) for r in old['geometry'] if r['split']=='discovery' and r['family']=='task'};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 weights={k:tuple(m.weight.double() for m in [b.mlp.Left,b.mlp.Right,b.mlp.Down]) for k,b in enumerate(model.blocks) if k>=8};axes={};torch.manual_seed(889)
 for axis in ['A','B']:
  for k in range(8,18):
   mean,basis,_=fits['agreement','task',axis,k];axes[axis,k]=(mean.cuda(),basis[0].cuda(),torch.randn(4,1152,device='cuda',dtype=torch.double))
 state={'capture':True};handles=[];normerr=0.;coorderr=0.;distanceerr=0.;infeasible=0;maxdelta=0.
 def hook(k):
  def f(mod,args,out):
   nonlocal normerr,coorderr,distanceerr,infeasible,maxdelta
   if state['capture'] or k not in SETS[state['subset']]:return
   mode=state['mode'];dose=state['dose'];seed=state['seed'];mean,b,random=axes[state['axis'],k];u=args[0][:,-1].double();p=u@b;c=p-mean@b;cent=cc[state['axis'],k];shift=dose*(cent[1-state['orders']]-c);pnew=p+shift;raw=u+shift[:,None]*b;norm2=u.square().sum(-1);rem=norm2-pnew.square()
   if mode=='identity':delta=torch.zeros_like(u)
   elif mode=='raw':delta=raw-u
   elif mode=='radial':delta=u*(raw.norm(dim=-1)/u.norm(dim=-1)-1)[:,None]
   else:
    infeasible+=int((rem<=0).sum());assert bool((rem>0).all());w=u-p[:,None]*b;sphere=pnew[:,None]*b+w*(rem/w.square().sum(-1)).sqrt()[:,None]
    if mode=='sphere':edited=sphere;coorderr=max(coorderr,float(((edited@b)-pnew).abs().max()))
    else:
     cosine=(sphere*u).sum(-1)/norm2;rand=random[seed].expand_as(u).clone();rand-=u*((rand*u).sum(-1)/norm2)[:,None];rand/=rand.norm(dim=-1,keepdim=True);edited=cosine[:,None]*u+(1-cosine.square()).clamp_min(0).sqrt()[:,None]*u.norm(dim=-1,keepdim=True)*rand;distanceerr=max(distanceerr,float(((edited-u).norm(dim=-1)-(sphere-u).norm(dim=-1)).abs().max()))
    normerr=max(normerr,float((edited.norm(dim=-1)-u.norm(dim=-1)).abs().max()));delta=edited-u
   maxdelta=max(maxdelta,float(delta.norm(dim=-1).max()));L,R,D=weights[k];change=((delta@L.T)*(u@R.T)+(u@L.T)*(delta@R.T)+(delta@L.T)*(delta@R.T))@D.T;z=out.clone();z[:,-1]+=change.to(out);return z
  return f
 for k,b in enumerate(model.blocks):
  if k>=8:handles.append(b.mlp.register_forward_hook(hook(k)))
 index={(e['pair'],e['format'],e['family'],e['order']):i for i,e in enumerate(ex)};buckets=defaultdict(list)
 for i in selection:buckets[len(ex[i]['ids'])].append(i)
 records=[];actual=0;replay=0.;identity=0.
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),12):
    ix=ii[st:st+12];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda');state['capture']=True;baseline=final_logits(model,tokens).double().cpu();actual+=len(ix);state['capture']=False;state['orders']=torch.tensor([ex[i]['order'] for i in ix],device='cuda')
    for n,i in enumerate(ix):replay=max(replay,float((baseline[n]-data['native'][i]).abs().max()))
    for axis in ['A','B']:
     state['axis']=axis
     for subset,mode,dose,seed in CASES:
      state.update(subset=subset,mode=mode,dose=dose,seed=seed);logits=final_logits(model,tokens).double().cpu();actual+=len(ix)
      for n,i in enumerate(ix):
       e=ex[i];y=logits[n];base=baseline[n];orient=1 if e['order']==0 else -1;ids=[v[0] for v in e['groups']];lp=torch.log_softmax(base,-1);lq=torch.log_softmax(y,-1);r={key:e[key] for key in ['pair','cohort','format','family','order','id']}|dict(axis=axis,subset=subset,mode=mode,dose=dose,seed=seed,margin=orient*float(y[ids[0]]-y[ids[1]]),native_margin=orient*float(base[ids[0]]-base[ids[1]]),top1_correct=int(y.argmax())==ids[e['order']],native_top1_correct=int(base.argmax())==ids[e['order']],KL=float((lp.exp()*(lp-lq)).sum()))
       for family,label in [('copy','noun'),('task','number')]:
        jj=index[e['pair'],e['format'],family,e['order']];gg=[v[0] for v in ex[jj]['groups']];r[label+'_margin']=orient*float(y[gg[0]]-y[gg[1]]);r['native_'+label+'_margin']=orient*float(base[gg[0]]-base[gg[1]])
       if mode=='identity':identity=max(identity,float((y-base).abs().max()))
       records.append(r)
   print('DONE',length,actual,flush=True)
 finally:
  for h in handles:h.remove()
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),native_replay_max_abs=replay,identity_max_abs=identity,sphere_norm_max_abs=normerr,sphere_coordinate_max_abs=coorderr,random_distance_match_max_abs=distanceerr,infeasible_sphere_targets=infeasible,max_local_delta_norm=maxdelta,whole_model_forwards=actual,expected_forwards=len(selection)*(1+2*len(CASES)),test_evaluations=0,finite=all(np.isfinite(r['margin']) and np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and max(replay,identity)<1e-4 and max(normerr,coorderr,distanceerr)<1e-9 and infeasible==0 and actual==audits['expected_forwards'] and audits['finite'],audits
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,selection=selection,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
