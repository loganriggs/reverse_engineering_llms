"""H891 discovery-only output-vector approximation for number-context interaction."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
import torch,numpy as np,json,time
from pathlib import Path
from collections import defaultdict
S='bilin18_number_interaction_compression';SETS={'centered':[12]}
CASES=[('centered',mode,1.,0) for mode in ['identity','exact_cross','mean','rank1','rank2','rank4','rank7','random7','wrong_mean','pooled_mean']]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'panel':ART/'bilin18_number_transfer_screen_captures.pt','selection':ART/'bilin18_number_transfer_screen_analysis.json','fits':ART/'bilin18_agreement_partner_number_fits.pt','centroids':ART/'bilin18_agreement_partner_number.json','discovery_states':ART/'bilin18_partner_subspace_fidelity_native_mlp_states.pt','discovery_fitmeta':ART/'bilin18_partner_subspace_fidelity.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H891','hashes':hashes,'sets':SETS,'cases':CASES,'design':'Fit centered cross-output-vector mean and PCA ranks1,2,4,7 on original8 discovery endpoints peraxis/family using matching old template. Bothaxes applied to all120 H887 eligible new task/copy contexts. Random7,wrongfamilymean,pooledmean controls. Compare against exact cross-only intervention via signed effects and KL(exactedited||approxedited). No rank selection on new examples; not wholemodel compression.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];eligible=json.loads(paths['selection'].read_text())['paired_competent_cells'];selection=[i for i,e in enumerate(ex) if [e['cohort'],e['format']] in eligible];assert len(selection)==120;fits=torch.load(paths['fits'],weights_only=False);old=json.loads(paths['centroids'].read_text());cc={(r['format'],r['layer']):torch.tensor(r['centroids'],device='cuda',dtype=torch.double) for r in old['geometry'] if r['split']=='discovery' and r['family']=='task'};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 weights={k:tuple(m.weight.double() for m in [b.mlp.Left,b.mlp.Right,b.mlp.Down]) for k,b in enumerate(model.blocks) if k>=8};axes={};torch.manual_seed(889)
 for axis in ['A','B']:
  for k in range(8,18):
   mean,basis,_=fits['agreement','task',axis,k];axes[axis,k]=(mean.cuda(),basis[0].cuda(),torch.randn(4,1152,device='cuda',dtype=torch.double))
 fitmeta=json.loads(paths['discovery_fitmeta'].read_text())['fitmeta'];disc=torch.load(paths['discovery_states'],weights_only=False);outputfits={};supports=[];orth=0.;trainrecon=0.;torch.manual_seed(891)
 for axis in ['A','B']:
  mean,b,_=axes[axis,12];origin=mean@b+cc[axis,12].mean();L,R,D=weights[12];Lb=b@L.T;Rb=b@R.T
  for family in ['copy','task']:
   info=next(r for r in fitmeta if (r['domain'],r['family'],r['format'],r['layer'])==('agreement',family,axis,12));jj=info['donors'];X=torch.stack([disc[i,(12,'raw')] for i in jj]).cuda();u=X/(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();w=u-((u@b)-origin)[:,None]*b;C=((w@L.T)*Rb+(w@R.T)*Lb)@D.T;mu=C.mean(0);_,sing,V=torch.linalg.svd(C-mu,full_matrices=False);basis=V[:7];rand=torch.linalg.qr(torch.randn(1152,7,device='cuda',dtype=torch.double),mode='reduced').Q.T;orth=max(orth,float((basis@basis.T-torch.eye(7,device='cuda')).abs().max()),float((rand@rand.T-torch.eye(7,device='cuda')).abs().max()));trainrecon=max(trainrecon,float(((C-mu)-(C-mu)@basis.T@basis).norm()/(C-mu).norm()));outputfits[axis,family]=(mu,basis,rand);supports.append({'axis':axis,'family':family,'donors':jj,'singular_values':sing.cpu().tolist()})
 state={'capture':True};handles=[];normerr=0.;coorderr=0.;distanceerr=0.;infeasible=0;maxdelta=0.;algebraerr=0.;fliperr=0.;moduleerr=0.;termvectors={};termnorms=[];exactlogps={};geometry=[]
 def hook(k):
  def f(mod,args,out):
   nonlocal algebraerr,fliperr,moduleerr
   if state['capture']:return
   mode=state['mode'];mean,b,random=axes[state['axis'],k];u=args[0][:,-1].double();p=u@b;c=p-mean@b;cent=cc[state['axis'],k];shift=cent[1-state['orders']]-c;origin=torch.zeros((),device='cuda',dtype=torch.double) if state['subset']=='absolute' else mean@b+cent.mean();t=p-origin;tp=t+shift;w=u-t[:,None]*b;L,R,D=weights[k];Lb=b@L.T;Rb=b@R.T;Q=(Lb*Rb)@D.T;C=((w@L.T)*Rb+(w@R.T)*Lb)@D.T;base=(w@L.T*(w@R.T))@D.T+mod.Down_bias.double();native=(u@L.T*(u@R.T))@D.T+mod.Down_bias.double();algebraerr=max(algebraerr,float((base+t[:,None]*C+t.square()[:,None]*Q-native).norm()/native.norm()));moduleerr=max(moduleerr,float((native-out[:,-1]).norm()/native.norm()))
   approx=[]
   for n,i in enumerate(state['indices']):
    family=ex[i]['family'];mu,basis,rand=outputfits[state['axis'],family];other=outputfits[state['axis'],'task' if family=='copy' else 'copy'][0]
    if mode in ['identity','exact_cross']:v=C[n]
    elif mode=='mean':v=mu
    elif mode=='wrong_mean':v=other
    elif mode=='pooled_mean':v=(mu+other)/2
    else:
     B=rand if mode=='random7' else basis[:int(mode[4:])];v=mu+(C[n]-mu)@B.T@B
    approx.append(v);geometry.append({'index':i,'axis':state['axis'],'mode':mode,'family':family,'relative_C_error':float((v-C[n]).norm()/C[n].norm())})
   change=shift[:,None]*torch.stack(approx) if mode!='identity' else torch.zeros_like(C)
   z=out.clone();z[:,-1]+=change.to(out);return z
  return f
 for k,b in enumerate(model.blocks):
  if k==12:handles.append(b.mlp.register_forward_hook(hook(k)))
 index={(e['pair'],e['format'],e['family'],e['order']):i for i,e in enumerate(ex)};buckets=defaultdict(list)
 for i in selection:buckets[len(ex[i]['ids'])].append(i)
 records=[];actual=0;replay=0.;identity=0.
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),12):
    ix=ii[st:st+12];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda');state['capture']=True;baseline=final_logits(model,tokens).double().cpu();actual+=len(ix);state['capture']=False;state['indices']=ix;state['orders']=torch.tensor([ex[i]['order'] for i in ix],device='cuda')
    for n,i in enumerate(ix):replay=max(replay,float((baseline[n]-data['native'][i]).abs().max()))
    for axis in ['A','B']:
     state['axis']=axis
     for subset,mode,dose,seed in CASES:
      state.update(subset=subset,mode=mode,dose=dose,seed=seed);logits=final_logits(model,tokens).double().cpu();actual+=len(ix)
      for n,i in enumerate(ix):
       e=ex[i];y=logits[n];base=baseline[n];orient=1 if e['order']==0 else -1;ids=[v[0] for v in e['groups']];lp=torch.log_softmax(base,-1);lq=torch.log_softmax(y,-1);r={key:e[key] for key in ['pair','cohort','format','family','order','id']}|dict(axis=axis,subset=subset,mode=mode,dose=dose,seed=seed,margin=orient*float(y[ids[0]]-y[ids[1]]),native_margin=orient*float(base[ids[0]]-base[ids[1]]),top1_correct=int(y.argmax())==ids[e['order']],native_top1_correct=int(base.argmax())==ids[e['order']],KL=float((lp.exp()*(lp-lq)).sum()))
       for family,label in [('copy','noun'),('task','number')]:
        jj=index[e['pair'],e['format'],family,e['order']];gg=[v[0] for v in ex[jj]['groups']];r[label+'_margin']=orient*float(y[gg[0]]-y[gg[1]]);r['native_'+label+'_margin']=orient*float(base[gg[0]]-base[gg[1]])
       if mode=='exact_cross':exactlogps[i,axis]=lq.clone()
       if mode!='identity':
        ref=exactlogps[i,axis];r['exact_to_approx_KL']=float((ref.exp()*(ref-lq)).sum());r['exact_to_native_KL']=float((ref.exp()*(ref-lp)).sum())
       if mode=='identity':identity=max(identity,float((y-base).abs().max()))
       records.append(r)
   print('DONE',length,actual,flush=True)
 finally:
  for h in handles:h.remove()
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),native_replay_max_abs=replay,identity_max_abs=identity,bilinear_decomposition_relative=algebraerr,basis_orthogonality_max_abs=orth,rank7_discovery_reconstruction_relative=trainrecon,double_formula_vs_module_relative=moduleerr,whole_model_forwards=actual,expected_forwards=len(selection)*(1+2*len(CASES)),test_evaluations=0,finite=all(np.isfinite(r['margin']) and np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and max(replay,identity)<1e-4 and algebraerr<1e-12 and orth<1e-10 and trainrecon<1e-10 and moduleerr<1e-5 and actual==audits['expected_forwards'] and audits['finite'],audits
 torch.save({key:tuple(v.cpu() for v in value) for key,value in outputfits.items()},ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,selection=selection,records=records,fit_supports=supports,geometry=geometry,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
