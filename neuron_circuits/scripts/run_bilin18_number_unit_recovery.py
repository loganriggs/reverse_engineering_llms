"""H894 disjoint lexical discovery recovery at fixed L12, rank1 only."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model,final_logits
from pathlib import Path
from collections import defaultdict
import json,time,itertools
import torch,numpy as np
S='bilin18_number_unit_recovery';SETS={'original':('old',[19,20,21,22]),'old_transfer':('old',[23,24,25,26]),'new_regular_1':('new',[8,9,10,11]),'new_regular_2':('new',[12,13,14,15]),'irregular':('new',[0,1,2,3])};MODES=['identity','replace','flip_exact','flip_replace']
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','old_states':ART/'bilin18_partner_subspace_fidelity_native_mlp_states.pt','old_map':ART/'bilin18_partner_subspace_fidelity.json','old_panel':ART/'bilin18_multidomain_matched_competence_captures.pt','new_panel':ART/'bilin18_number_transfer_screen_captures.pt','evaluation':ART/'bilin18_blind_number_unit_manifest.json'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H894','hashes':hashes,'discovery_sets':SETS,'formats':['A','B'],'modes':MODES,'design':'Five disjoint4-noun discovery sets, each8 endpoints per template. Sameunsupervised first inputPC and outputPC, fixedL12. Orient numberaxis bydiscovery labels and fitcentroids, no PC selection ifnumberclassificationfails. Evaluate all10 fits on same64 H893 synthetic contexts, nofamily fitrouting. Remaining384 naturaldocs untouched inthisrun; fits frozen forsubsequent evaluation.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');oldpanel=torch.load(paths['old_panel'],weights_only=False);oldmap=json.loads(paths['old_map'].read_text());oldex=[oldpanel['examples'][i] for i in oldmap['selection_original_indices']];oldstates=torch.load(paths['old_states'],weights_only=False);new=torch.load(paths['new_panel'],weights_only=False);examples=[e for e in json.loads(paths['evaluation'].read_text())['examples'] if e['group']!='natural'];assert len(examples)==64;model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 mod=model.blocks[12].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];eps=torch.finfo(torch.float32).eps;units={};fitinfo=[];orth=0.;compileerr=0.
 for name,(source,pairs) in SETS.items():
  ex=oldex if source=='old' else new['examples']
  for fmt in ['A','B']:
   ix=[i for i,e in enumerate(ex) if e['pair'] in pairs and e['family']=='task' and e['format']==fmt];assert len(ix)==8;X=torch.stack([oldstates[i,(12,'raw')] if source=='old' else new['states'][i][12] for i in ix]).cuda();u=X/(X.square().mean(-1,keepdim=True)+eps).sqrt();mean=u.mean(0);_,sing,V=torch.linalg.svd(u-mean,full_matrices=False);b=V[0];labels=torch.tensor([ex[i]['order'] for i in ix],device='cuda');coords=(u-mean)@b;cc=torch.stack([coords[labels==o].mean() for o in [0,1]])
   if cc[1]<cc[0]:b=-b;coords=-coords;cc=-cc
   origin=mean@b+cc.mean();t=u@b-origin;w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;mu=C.mean(0);_,csing,CV=torch.linalg.svd(C-mu,full_matrices=False);v=CV[0];back=D.T@v;a=L.T@((R@b)*back)+R.T@((L@b)*back);encoder=a-(a@b)*b;offset=origin*(a@b)-mu@v;h=mu+offset*v;Q=((b@L.T)*(b@R.T))@D.T;key=name+'_'+fmt;units[key]=dict(b=b,origin=origin,cc=cc,mean_projection=mean@b,mu=mu,v=v,encoder=encoder,offset=offset,Q=Q,h=h);orth=max(orth,abs(float(b.norm()-1)),abs(float(v.norm()-1)));fitinfo.append(dict(unit=key,source=source,pairs=pairs,indices=ix,number_training_accuracy=float(((coords>cc.mean()).long()==labels).double().mean()),input_PC_variance_fraction=float(sing[0].square()/sing.square().sum()),output_PC_variance_fraction=float(csing[0].square()/csing.square().sum())))
 state={'mode':'baseline'};stash={};writevectors={};prediction=[]
 def hook(mod,args,out):
  nonlocal compileerr
  u=args[0][:,-1].double();stash['u']=u.cpu()
  if state['mode']=='baseline':return
  unit=units[state['unit']];b=unit['b'];t=u@b-unit['origin'];w=u-t[:,None]*b;C=((w@L.T)*(b@R.T)+(w@R.T)*(b@L.T))@D.T;hat=unit['mu']+(u@unit['encoder']+unit['offset'])[:,None]*unit['v'];ref=unit['mu']+((C-unit['mu'])@unit['v'])[:,None]*unit['v'];compileerr=max(compileerr,float((hat-ref).norm()/ref.norm()));mode=state['mode'];target=unit['cc'][1-state['orders']]+unit['mean_projection']-unit['origin'];tp=target if mode.startswith('flip') else t;chat=hat if mode in ['replace','flip_replace'] else C;change=tp[:,None]*chat-t[:,None]*C+(tp.square()-t.square())[:,None]*unit['Q'];z=out.clone();z[:,-1]+=change.to(out)
  if mode=='identity':
   for n,i in enumerate(state['indices']):writevectors[i,state['unit']]=(t[n]*hat[n]).cpu();prediction.append(dict(index=i,unit=state['unit'],group=examples[i]['group'],number_correct=int(t[n]>0)==examples[i]['order']))
  return z
 handle=mod.register_forward_hook(hook);buckets=defaultdict(list)
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 records=[];actual=0;identity=0.;refs={}
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),8):
    ix=ii[st:st+8];tokens=torch.tensor([examples[i]['ids'] for i in ix],device='cuda');state.update(mode='baseline',indices=ix,orders=torch.tensor([examples[i]['order'] for i in ix],device='cuda'));base=final_logits(model,tokens).double().cpu();actual+=len(ix)
    for key in units:
     state['unit']=key
     for mode in MODES:
      state['mode']=mode;y=final_logits(model,tokens).double().cpu();actual+=len(ix)
      for n,i in enumerate(ix):
       e=examples[i];lp=torch.log_softmax(base[n],-1);lq=torch.log_softmax(y[n],-1);ids=e['answer_ids'];sign=1 if e['order']==0 else -1
       if mode=='identity':identity=max(identity,float((y[n]-base[n]).abs().max()))
       if mode=='flip_exact':refs[i,key]=lq.clone()
       r=dict(index=i,group=e['group'],pair=e['pair'],format=e['format'],order=e['order'],unit=key,mode=mode,margin=sign*float(y[n,ids[0]]-y[n,ids[1]]),native_margin=sign*float(base[n,ids[0]]-base[n,ids[1]]),KL=float((lp.exp()*(lp-lq)).sum()),top1_changed=int(base[n].argmax())!=int(y[n].argmax()))
       if mode=='flip_replace':ref=refs[i,key];r['exact_flip_to_approx_KL']=float((ref.exp()*(ref-lq)).sum())
       records.append(r)
   print('DONE',length,actual,flush=True)
 finally:handle.remove()
 comparisons=[]
 for x,y in itertools.combinations(units,2):
  X,Y=units[x],units[y];matrixcos=float((X['v']@Y['v'])*(X['encoder']@Y['encoder'])/(X['encoder'].norm()*Y['encoder'].norm()));aa=torch.stack([writevectors[i,x] for i in range(64)]);bb=torch.stack([writevectors[i,y] for i in range(64)]);comparisons.append(dict(first=x,second=y,number_axis_cosine=float(X['b']@Y['b']),output_direction_abs_cosine=abs(float(X['v']@Y['v'])),compiled_rank1_matrix_cosine=matrixcos,native_cross_write_relative_difference=float((aa-bb).norm()/aa.norm())))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),identity_logits_max_abs=identity,unit_norm_max_abs=orth,compiled_encoder_relative=compileerr,whole_model_forwards=actual,expected_forwards=64*(1+10*4),test_evaluations=0,finite=all(np.isfinite(r['KL']) for r in records));assert audits['hashes_unchanged'] and identity<1e-4 and orth<1e-10 and compileerr<1e-10 and actual==audits['expected_forwards'] and audits['finite'];torch.save({k:{n:v.cpu() for n,v in u.items()} for k,u in units.items()},ART/f'{S}_units.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,fitinfo=fitinfo,examples=examples,records=records,geometry=comparisons,predictions=prediction,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
