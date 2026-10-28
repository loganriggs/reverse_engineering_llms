"""H938 fresh-panel live evaluation of discovery-CV frozen query models."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model,final_logits
from run_bilin18_discovery_query_crossvalidation import predict
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
S='bilin18_fresh_query_model_fidelity';HEADS=[(8,3),(13,0),(14,4)];MODES=['native_routing','zero','distance_mean','linear0']+[f'hybrid{k}s{s}' for k in [0,1] for s in [0,1,2]]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'panel':ART/'bilin18_frozen_query_fresh_panel_captures.pt','baseline':ART/'bilin18_multidomain_coefficient_prediction_fits.pt','predictor':ROOT/'scripts/run_bilin18_discovery_query_crossvalidation.py','frozen':ART/'bilin18_discovery_query_crossvalidation_fits.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H938','hashes':hashes,'heads':HEADS,'modes':MODES,'design':'No fitting or selection: evaluate all21 frozen H935 head-models in joint3head live interventions. linear0 uses selected linear; hybrid0s{seed} learnedkmeans; hybrid1s{seed} randomhyperplanes. Each head uses own discovery-CV alpha. Preserve native current values; replace persistent component with predicted routing and native/donor bus. Compare full output KL and donor-effect preservation on H937 fresh384 rows with new vocabulary and familiar/new formats; all competence cells pass. Unseen end-distances predict zero for all models, as frozen in H937. No reserved test. Random groups are geometric and may contain context; capacities/occupancies not identical.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];ix=list(range(len(ex)));assert len(ix)==384;lookup={(e['id'],e['order']):i for i,e in enumerate(ex)};donors={i:lookup[ex[i]['id'],1-ex[i]['order']] for i in ix};fits=torch.load(paths['baseline'],weights_only=False)['fits'];pred={}
 screen=json.loads((ART/'bilin18_frozen_query_fresh_panel.json').read_text());assert screen['audits']['source_vocabulary_disjoint_from_H882'] and all(c['pass'] for c in screen['competence'])
 for i in ix:
  T=len(ex[i]['ids'])
  for l,h in HEADS:
   pred[i,l,h,'zero']=torch.zeros(T,dtype=torch.double)
   pred[i,l,h,'distance_mean']=torch.tensor([fits.get(('distance_mean',l,h,T-1-p),0.) for p in range(T)],dtype=torch.double)
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 frozen=torch.load(paths['frozen'],weights_only=False);train=frozen['discovery_indices'];frozenfits=frozen['fits'];assert all(f['train_indices']==train for f in frozenfits.values());queries={};gatelog=[];fallback=0;normalerr=max(f['normal_error'] for f in frozenfits.values())
 state={'mode':'native_routing','donor':False,'batch':[]};live={};handles=[];originals={}
 for l,h in HEADS:
  for which in ['q','q2']:
   def cap(mod,args,out,l=l,h=h,which=which):
    z=out[:,-1].reshape(-1,9,128)[:,h];queries[l,h,which]=torch.nn.functional.rms_norm(z,(128,)).double()
   handles.append(getattr(model.blocks[l].attn,'c_'+which).register_forward_hook(cap))
  fn=model.blocks[l].attn.squared_attention;originals[l]=fn
  def capture(q,k,v,q2,k2,l=l,h=h,fn=fn):
   live[l,h]=((torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128))[:,h];return fn(q,k,v,q2,k2)
  model.blocks[l].attn.squared_attention=capture
  def edit(mod,args,l=l,h=h):
   nonlocal fallback
   if state['mode']=='native_routing' and not state['donor']:return
   dd=[]
   for n,i in enumerate(state['batch']):
    A=live[l,h][n].double();mode=state['mode']
    if mode.startswith(('linear','hybrid')):
     q=torch.cat([queries[l,h,'q'][n],queries[l,h,'q2'][n]]).cpu()/16
     if mode.startswith('linear'):kind,seed='linear',-1
     else:
      code,seed=map(int,mode[6:].split('s'));kind=['kmeans','random'][code]
     f=frozenfits[l,h,kind,seed];known=len(f['positions']);missing=max(0,len(A)-known);fallback+=missing;B=torch.cat([torch.zeros(missing,dtype=torch.double),predict(q,min(len(A),known),f)]).cuda()
    else:B=A if mode=='native_routing' else pred[i,l,h,mode].cuda()
    V=data['bus'][i][:,h].double().cuda();j=donors[i] if state['donor'] else i;W=data['bus'][j][:,h].double().cuda();dd.append(float(model.blocks[l].attn.lamb)*(B[:,None]*W-A[:,None]*V).sum(0))
   x=args[0].clone();x[:,-1,h*128:(h+1)*128]+=torch.stack(dd).to(x);return (x,)
  handles.append(model.blocks[l].attn.c_proj.register_forward_pre_hook(edit))
 buckets=defaultdict(list)
 for i in ix:buckets[len(ex[i]['ids'])].append(i)
 records=[];actual=0;replay=0.;identity=0.
 try:
  for length,inds in sorted(buckets.items()):
   for st in range(0,len(inds),8):
    batch=inds[st:st+8];state.update(mode='native_routing',donor=False,batch=batch);tokens=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');native=final_logits(model,tokens).double().cpu();actual+=len(batch)
    for n,i in enumerate(batch):replay=max(replay,float((native[n]-data['native'][i]).abs().max()))
    yy={}
    for mode in MODES:
     for donor in [False,True]:
      state.update(mode=mode,donor=donor);yy[mode,donor]=final_logits(model,tokens).double().cpu();actual+=len(batch)
    identity=max(identity,float((native-yy['native_routing',False]).abs().max()))
    for mode in MODES:
     for n,i in enumerate(batch):
      e=ex[i];v=e['groups'][e['expected']][0];u=e['groups'][1-e['expected']][0];margin=lambda z:float(z[v]-z[u]);z=yy[mode,False][n];w=yy[mode,True][n];ref=yy['native_routing',True][n];lp=native[n].log_softmax(-1);lz=z.log_softmax(-1);lf=ref.log_softmax(-1);lw=w.log_softmax(-1)
      records.append({k:e[k] for k in ['domain','family','format','pair','order']}|{'index':i,'mode':mode,'native_KL':float((lp.exp()*(lp-lz)).sum()),'edited_KL':float((lf.exp()*(lf-lw)).sum()),'native_margin_error':margin(z)-margin(native[n]),'reference_effect':margin(ref)-margin(native[n]),'predicted_effect':margin(w)-margin(z),'effect_error':margin(w)-margin(z)-margin(ref)+margin(native[n]),'native_top1':int(native[n].argmax())==v,'predicted_top1':int(z.argmax())==v})
    print('DONE',length,st,actual,flush=True)
 finally:
  for handle in handles:handle.remove()
  for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
 grouped=defaultdict(list)
 for r in records:grouped[r['domain'],r['family'],r['format'],r['mode']].append(r)
 summary=[]
 for key,rr in grouped.items():
  rms=lambda k:float(np.mean([r[k]**2 for r in rr])**.5);summary.append(dict(zip(['domain','family','format','mode'],key))|{'native_KL':float(np.mean([r['native_KL'] for r in rr])),'edited_KL':float(np.mean([r['edited_KL'] for r in rr])),'native_margin_RMSE':rms('native_margin_error'),'reference_effect_RMS':rms('reference_effect'),'effect_RMSE':rms('effect_error'),'effect_error_ratio':rms('effect_error')/max(rms('reference_effect'),1e-12),'native_top1':float(np.mean([r['native_top1'] for r in rr])),'predicted_top1':float(np.mean([r['predicted_top1'] for r in rr]))})
 audits={'frozen_fit_sha256':hashes['frozen'],'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_max_abs':identity,'native_replay_max_abs':replay,'fit_eval_disjoint':screen['audits']['source_vocabulary_disjoint_from_H882'],'fit_indices':train,'evaluation_indices':ix,'whole_model_forwards':actual,'expected_forwards':len(ix)*(1+2*len(MODES)),'prototype_fallback_positions':fallback,'ridge_normal_equation_max_abs':normalerr,'finite':all(np.isfinite(r['native_KL']) and np.isfinite(r['effect_error']) for r in records),'test_evaluations':0}
 assert audits['hashes_unchanged'] and identity==0 and replay<1e-4 and audits['fit_eval_disjoint'] and actual==audits['expected_forwards'] and audits['finite'],audits
 (ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'gate_log':gatelog,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(summary,audits,flush=True)
if __name__=='__main__':main()
