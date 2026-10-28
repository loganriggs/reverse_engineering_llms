"""H888 frozen number-coordinate causal transfer: response versus native MLP."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits,partner_jvp
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
S='bilin18_frozen_number_causal';SETS={'core':list(range(11,17)),'complement':[8,9,10,17]};MODES=['direct_identity','direct_correct','direct_flip','direct_erase','direct_random','response_self','response_correct','response_flip']
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'panel':ART/'bilin18_number_transfer_screen_captures.pt','selection':ART/'bilin18_number_transfer_screen_analysis.json','fits':ART/'bilin18_agreement_partner_number_fits.pt','centroids':ART/'bilin18_agreement_partner_number.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H888','hashes':hashes,'sets':SETS,'modes':MODES,'axes':['A','B'],'design':'All originally gate-eligible H887 task contexts; frozen axes/centroids. Bothaxes alltemplates, includingC. CoreL11–16 vscomplement. Direct normalized MLP-input coordinate clamp/erase/flip, plus matched-length orthogonal random shift. Separate response-Jacobian manipulation withnative offsets, symmetric L8H3 writes. No refit; no reservedtest.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];eligible=json.loads(paths['selection'].read_text())['paired_competent_cells'];selection=[i for i,e in enumerate(ex) if e['family']=='task' and [e['cohort'],e['format']] in eligible];assert len(selection)==60;fits=torch.load(paths['fits'],weights_only=False);old=json.loads(paths['centroids'].read_text());cc={(r['format'],r['layer']):torch.tensor(r['centroids'],dtype=torch.double,device='cuda') for r in old['geometry'] if r['split']=='discovery' and r['family']=='task'};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 weights={k:tuple(t.detach().double() for t in [b.mlp.Left.weight,b.mlp.Right.weight,b.mlp.Down.weight,b.mlp.Down_bias]) for k,b in enumerate(model.blocks) if k>=8};axes={};torch.manual_seed(888)
 for axis in ['A','B']:
  for k in range(8,18):
   mean,basis,_=fits['agreement','task',axis,k];b=basis[0].cuda();rand=torch.randn(1152,device='cuda',dtype=torch.double);rand-=b*(rand@b);rand/=rand.norm();axes[axis,k]=(mean.cuda(),b,rand)
 state={'capture':True};raw={};cache={};handles=[];eps=torch.finfo(torch.float32).eps;ortherr=0.;normerr=0.;coorderr=0.
 def pre(k):
  def f(mod,args):raw[k]=mod.lambdas[0]*args[0][:,-1]+mod.lambdas[1]*args[2][:,-1]
  return f
 def att(k):
  def f(mod,args,out):raw[k]=raw[k]+out[0][:,-1]
  return f
 def mlp(k):
  def f(mod,args,out):
   nonlocal ortherr,normerr,coorderr
   if state['capture']:cache[k]=(raw[k].double().clone(),out[:,-1].clone());return
   if k not in SETS[state['set']]:return
   mode=state['mode'];mean,b,rand=axes[state['axis'],k];cent=cc[state['axis'],k];native,y0=cache[k];u=native/(native.square().mean(-1,keepdim=True)+eps).sqrt() if mode.startswith('response') else args[0][:,-1].double();c=(u-mean)@b;correct=cent[state['orders']];opposite=cent[1-state['orders']]
   target=c if mode in ['direct_identity','response_self'] else cent.mean().expand_as(c) if mode=='direct_erase' else opposite if mode in ['direct_flip','response_flip','direct_random'] else correct
   shift=target-c;delta=shift[:,None]*(rand if mode=='direct_random' else b)
   normerr=max(normerr,float((delta.norm(dim=-1)-shift.abs()).abs().max()));ortherr=max(ortherr,abs(float(b@rand)))
   if mode!='direct_random':coorderr=max(coorderr,float((((u+delta-mean)@b)-target).abs().max()))
   if mode.startswith('response'):new=y0+partner_jvp(native,u+delta,raw[k].double()-native,weights[k]).to(y0)
   else:
    L,R,D,bias=weights[k];change=((delta@L.T)*(u@R.T)+(u@L.T)*(delta@R.T)+(delta@L.T)*(delta@R.T))@D.T;new=out[:,-1]+change.to(out)
   z=out.clone();z[:,-1]=new;return z
  return f
 for k,b in enumerate(model.blocks):
  if k>=8:handles.extend([b.register_forward_pre_hook(pre(k)),b.attn.register_forward_hook(att(k)),b.mlp.register_forward_hook(mlp(k))])
 index={(e['pair'],e['format'],e['family'],e['order']):i for i,e in enumerate(ex)};buckets=defaultdict(list)
 for i in selection:buckets[len(ex[i]['ids'])].append(i)
 records=[];actual=0;identityerr=0.;replay=0.;formulaerr=0.;ampmin=1e9
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),12):
    ix=ii[st:st+12];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda');state['capture']=True;baseline=final_logits(model,tokens).double().cpu();actual+=len(ix);state['capture']=False;state['orders']=torch.tensor([ex[i]['order'] for i in ix],device='cuda');attn=model.blocks[8].attn;original=attn.squared_attention;db=[];amps=[]
    for n,i in enumerate(ix):
     e=ex[i];replay=max(replay,float((baseline[n]-data['native'][i]).abs().max()));j=index[e['pair'],e['format'],'task',1-e['order']];p=e['source_positions'][0];db.append((data['bus'][j][p,3]-data['bus'][i][p,3]).double());ci=index[e['pair'],e['format'],'copy',e['order']];gamma=lambda t:float(attn.lamb)/2*float(data['routing'][t][ex[t]['source_positions']].sum());amps.append(float(np.sqrt(abs(gamma(i)*gamma(ci)))))
    ampmin=min(ampmin,min(amps));delta=torch.stack(db).cuda()*torch.tensor(amps,device='cuda',dtype=torch.double)[:,None]
    for axis in ['A','B']:
     state['axis']=axis
     for subset in SETS:
      state['set']=subset
      for mode in MODES:
       state['mode']=mode
       for sign in ([0,1,-1] if mode.startswith('response') else [0]):
        def edit(q,k,v,q2,k2):
         y=original(q,k,v,q2,k2)
         if sign:y=y.clone();y[:,3,-1]+=(sign*delta).to(y)
         return y
        attn.squared_attention=edit
        try:logits=final_logits(model,tokens).double().cpu()
        finally:attn.squared_attention=original
        actual+=len(ix)
        for n,i in enumerate(ix):
         e=ex[i];y=logits[n];base=baseline[n];lp=torch.log_softmax(base,-1);lq=torch.log_softmax(y,-1);ids=[v[0] for v in e['groups']];margin=float(y[ids[0]]-y[ids[1]]);native_margin=float(base[ids[0]]-base[ids[1]])
         if mode=='direct_identity' or (mode.startswith('response') and sign==0):identityerr=max(identityerr,float((y-base).abs().max()))
         records.append({key:e[key] for key in ['pair','cohort','format','order','id']}|dict(axis=axis,subset=subset,mode=mode,sign=sign,amplitude=amps[n],margin=margin,native_margin=native_margin,expected_margin=(1 if e['order']==0 else -1)*margin,native_expected_margin=(1 if e['order']==0 else -1)*native_margin,top1_correct=int(y.argmax())==ids[e['order']],native_top1_correct=int(base.argmax())==ids[e['order']],KL=float((lp.exp()*(lp-lq)).sum())))
   print('DONE',length,actual,flush=True)
 finally:
  for h in handles:h.remove()
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),native_replay_logits_max_abs=replay,identity_logits_max_abs=identityerr,coordinate_target_max_abs=coorderr,matched_delta_norm_max_abs=normerr,random_axis_orthogonality_max_abs=ortherr,min_amplitude=ampmin,whole_model_forwards=actual,test_evaluations=0,finite=all(np.isfinite(r['margin']) and np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and replay<1e-4 and identityerr<1e-4 and coorderr<1e-10 and normerr<1e-10 and ortherr<1e-10 and actual==3420 and audits['finite'],audits
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,selection=selection,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
