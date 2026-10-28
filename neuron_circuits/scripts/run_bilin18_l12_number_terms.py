"""H890 L12 exact bilinear number-coordinate terms with origin controls."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
import torch,numpy as np,json,time
from pathlib import Path
from collections import defaultdict
S='bilin18_l12_number_terms';SETS={'absolute':[12],'centered':[12]}
CASES=[(origin,mode,1.,0) for origin in SETS for mode in ['identity','full','coordinate_cross','coordinate_square','local_linear','local_quadratic']]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'panel':ART/'bilin18_number_transfer_screen_captures.pt','selection':ART/'bilin18_number_transfer_screen_analysis.json','fits':ART/'bilin18_agreement_partner_number_fits.pt','centroids':ART/'bilin18_agreement_partner_number.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H890','hashes':hashes,'sets':SETS,'cases':CASES,'design':'All120 eligible task/copy contexts; both frozen axes. L12 only. Decompose coordinate flip under absolute origin and discovery centroid midpoint; separately intervene cross and square output changes. Also native-Jacobian displacement and local quadratic remainder. Exact algebra/replay checks; no refit.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];eligible=json.loads(paths['selection'].read_text())['paired_competent_cells'];selection=[i for i,e in enumerate(ex) if [e['cohort'],e['format']] in eligible];assert len(selection)==120;fits=torch.load(paths['fits'],weights_only=False);old=json.loads(paths['centroids'].read_text());cc={(r['format'],r['layer']):torch.tensor(r['centroids'],device='cuda',dtype=torch.double) for r in old['geometry'] if r['split']=='discovery' and r['family']=='task'};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 weights={k:tuple(m.weight.double() for m in [b.mlp.Left,b.mlp.Right,b.mlp.Down]) for k,b in enumerate(model.blocks) if k>=8};axes={};torch.manual_seed(889)
 for axis in ['A','B']:
  for k in range(8,18):
   mean,basis,_=fits['agreement','task',axis,k];axes[axis,k]=(mean.cuda(),basis[0].cuda(),torch.randn(4,1152,device='cuda',dtype=torch.double))
 state={'capture':True};handles=[];normerr=0.;coorderr=0.;distanceerr=0.;infeasible=0;maxdelta=0.;algebraerr=0.;fliperr=0.;moduleerr=0.;termvectors={};termnorms=[]
 def hook(k):
  def f(mod,args,out):
   nonlocal algebraerr,fliperr,moduleerr
   if state['capture']:return
   mode=state['mode'];mean,b,random=axes[state['axis'],k];u=args[0][:,-1].double();p=u@b;c=p-mean@b;cent=cc[state['axis'],k];shift=cent[1-state['orders']]-c;origin=torch.zeros((),device='cuda',dtype=torch.double) if state['subset']=='absolute' else mean@b+cent.mean();t=p-origin;tp=t+shift;w=u-t[:,None]*b;L,R,D=weights[k];Lb=b@L.T;Rb=b@R.T;Q=(Lb*Rb)@D.T;C=((w@L.T)*Rb+(w@R.T)*Lb)@D.T;base=(w@L.T*(w@R.T))@D.T+mod.Down_bias.double();native=(u@L.T*(u@R.T))@D.T+mod.Down_bias.double();algebraerr=max(algebraerr,float((base+t[:,None]*C+t.square()[:,None]*Q-native).norm()/native.norm()));moduleerr=max(moduleerr,float((native-out[:,-1]).norm()/native.norm()))
   cross=shift[:,None]*C;square=(tp.square()-t.square())[:,None]*Q;local_linear=shift[:,None]*(((u@L.T)*Rb+(u@R.T)*Lb)@D.T);local_quadratic=shift.square()[:,None]*Q;full=cross+square;fliperr=max(fliperr,float((full-local_linear-local_quadratic).norm()/full.norm()))
   changes={'identity':torch.zeros_like(full),'full':full,'coordinate_cross':cross,'coordinate_square':square,'local_linear':local_linear,'local_quadratic':local_quadratic};change=changes[mode]
   if mode=='full':
    for n,i in enumerate(state['indices']):termvectors[i,state['axis'],state['subset']]={'cross':cross[n].cpu(),'square':square[n].cpu(),'local_linear':local_linear[n].cpu(),'local_quadratic':local_quadratic[n].cpu(),'full':full[n].cpu()}
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
       if mode=='identity':identity=max(identity,float((y-base).abs().max()))
       records.append(r)
   print('DONE',length,actual,flush=True)
 finally:
  for h in handles:h.remove()
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),native_replay_max_abs=replay,identity_max_abs=identity,bilinear_decomposition_relative=algebraerr,flip_decomposition_relative=fliperr,double_formula_vs_module_relative=moduleerr,whole_model_forwards=actual,expected_forwards=len(selection)*(1+2*len(CASES)),test_evaluations=0,finite=all(np.isfinite(r['margin']) and np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and max(replay,identity)<1e-4 and algebraerr<1e-12 and fliperr<1e-12 and moduleerr<1e-5 and actual==audits['expected_forwards'] and audits['finite'],audits
 torch.save(termvectors,ART/f'{S}_term_vectors.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,selection=selection,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
