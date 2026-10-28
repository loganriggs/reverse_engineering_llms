"""H876 final RMS denominator interventions with all earlier computation live."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_broad_task_competence import continuation_id,final_logits
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_final_scale_clamp';TARGETS=[(13,0),(14,4),(8,3)]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'panel':ART/'bilin18_reader_recombination_panel_captures.pt','screen':ART/'bilin18_reader_recombination_panel.json','live':ART/'bilin18_reader_gain_confirmation.json','finals':ART/'bilin18_reader_normalization_control_final_states.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H876','hashes':hashes,'design':'All192H873 endpoints,3frozen readers,identity andsamecommon +/-headwrites asH874. Captureactual final residual afterlive downstream computation. Intervene only finalreadout denominator; no refit.','denominators':'native=RMS(editedstate);own_fixed=native unedited recipientRMS;shared_fixed=geometricmean nativecopy/capitalRMS;shared_dynamic=RMS(editedstate)*shared_fixed/native_recipientRMS, preserving perturbation-induced radialfeedback. Extra shared_fixed_no_cap diagnostic removes30tanh. All readoutvariants evaluated exactly on same editedstates.','scope':'Counterfactual outputnormalization interventions, not naturally realized modelstates. NewAB/C competent,capitalD failedcontrol retained. Sameformat/countrypair/endpoint matched across tasks;commoninputwrite vector fixed. No reservedtest.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];finals=torch.load(paths['finals'],weights_only=False)['new'].double();eps=torch.finfo(torch.float32).eps;radii=(finals.square().mean(1)+eps).sqrt();live=json.loads(paths['live'].read_text());previous={(r['id'],r['order'],r['layer'],r['head'],r['mode']):r for r in live['records']};screen=json.loads(paths['screen'].read_text());tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);idsby={}
 for i,e in enumerate(ex):idsby[i]={typ:[continuation_id(tok,e['text'],' '+v[part]) for v in screen['config']['pairs'][e['pair']]] for typ,part in [('country',0),('capital',1)]}
 index={(e['id'],e['order']):i for i,e in enumerate(ex)};matched={(e['pair'],e['format'],e['order'],e['family']):i for i,e in enumerate(ex)};shared=[]
 for i,e in enumerate(ex):other=matched[e['pair'],e['format'],e['order'],'capital' if e['family']=='copy' else 'copy'];shared.append((radii[i]*radii[other]).sqrt())
 shared=torch.stack(shared);assert all(float(shared[i])==float(shared[matched[e['pair'],e['format'],e['order'],'capital' if e['family']=='copy' else 'copy']]) for i,e in enumerate(ex));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 U=model.lm_head.weight.double().cpu();bias=None if model.lm_head.bias is None else model.lm_head.bias.double().cpu();buckets=defaultdict(list)
 for i,e in enumerate(ex):buckets[len(e['ids'])].append(i)
 stash={}
 def hook(mod,args,out):stash['final']=out[0][:,-1].double().cpu()
 handle=model.blocks[-1].register_forward_hook(hook);records=[];captures={};actual=0;nativeerr=0.;marginerr=0.;readouterr=0.;finalerr=0.;shared_identity_err=0.
 try:
  for length,ii in sorted(buckets.items()):
   for st in range(0,len(ii),12):
    ix=ii[st:st+12];tokens=torch.tensor([ex[i]['ids'] for i in ix],device='cuda')
    for l,h in TARGETS:
     att=model.blocks[l].attn;original=att.squared_attention;db=[];amps=[]
     for i in ix:
      e=ex[i];j=index[e['id'],1-e['order']];p=e['source_positions'][0];db.append((data['bus'][j][p,h]-data['bus'][i][p,h]).double());amps.append(previous[e['id'],e['order'],l,h,'common_positive']['routing_coefficient'])
     delta=torch.stack(db).cuda()*torch.tensor(amps,device='cuda',dtype=torch.double)[:,None]
     for sign in [0,1,-1]:
      def edit(q,k,v,q2,k2):
       y=original(q,k,v,q2,k2)
       if sign:y=y.clone();y[:,h,-1]+=(sign*delta).to(y)
       return y
      att.squared_attention=edit
      try:y=final_logits(model,tokens)
      finally:att.squared_attention=original
      actual+=len(ix);states=stash['final'];
      for k,i in enumerate(ix):
       e=ex[i];z=states[k];r=(z.square().mean()+eps).sqrt();captures[i,l,h,sign]=z
       if sign==0:nativeerr=max(nativeerr,float((y[k].double().cpu()-data['native'][i]).abs().max()));finalerr=max(finalerr,float((z-finals[i]).abs().max()));shared_identity_err=max(shared_identity_err,float(abs(r*shared[i]/radii[i]-shared[i])))
       for typ,outids in idsby[i].items():
        w=U[outids];b=torch.zeros(2,dtype=torch.double) if bias is None else bias[outids];denoms={'native':r,'own_fixed':radii[i],'shared_fixed':shared[i],'shared_dynamic':r*shared[i]/radii[i],'shared_fixed_no_cap':shared[i]}
        for mode,den in denoms.items():
         logits=w@(z/den)+b
         if mode!='shared_fixed_no_cap':logits=30*torch.tanh(logits/30)
         margin=float(logits[0]-logits[1])
         if mode=='native':
          readouterr=max(readouterr,float((logits-y[k].double().cpu()[outids]).abs().max()))
          if sign:old=previous[e['id'],e['order'],l,h,'common_positive' if sign==1 else 'common_negative'];marginerr=max(marginerr,abs(margin-old['edited_'+typ+'_margin']))
         records.append({key:e[key] for key in ['pair','format','family','order','id']}|{'layer':l,'head':h,'sign':sign,'amplitude':amps[k],'output':typ,'normalization':mode,'margin':margin,'native_RMS':float(radii[i]),'edited_RMS':float(r),'denominator':float(den)})
   print('DONE',length,actual,flush=True)
 finally:handle.remove()
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_logits_max_abs':nativeerr,'native_finalstate_replay_max_abs':finalerr,'readout_logits_max_abs':readouterr,'prior_commonwrite_margin_max_abs':marginerr,'shared_denominator_equal':True,'shared_dynamic_identity_max_abs':shared_identity_err,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['margin']) for r in records)};assert audits['hashes_unchanged'] and max(nativeerr,readouterr,marginerr)<1e-4 and finalerr<1e-4 and shared_identity_err<1e-8 and actual==192*3*3 and audits['finite'];torch.save(captures,ART/f'{S}_final_states.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
