"""H875 parameter-free downstream normalization and frozen-update controls."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_broad_task_competence import continuation_id,final_logits
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_reader_normalization_control';TARGETS=[(13,0),(14,4),(8,3)]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'old_panel':ART/'bilin18_aligned_lexical_continuations_captures.pt','old_screen':ART/'bilin18_aligned_lexical_continuations.json','old_live':ART/'bilin18_reader_coefficient_interchange.json','new_panel':ART/'bilin18_reader_recombination_panel_captures.pt','new_screen':ART/'bilin18_reader_recombination_panel.json','new_live':ART/'bilin18_reader_gain_confirmation.json','gains':ART/'bilin18_reader_coefficient_interchange_canonical_responses.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H875','hashes':hashes,'design':'Nativefinalresidual capture on allH870/H873contexts. No fit. Compare normratio and exactfrozen-update readout with existing live symmetricheadwrites. Keep old/newformats andfailedcells separate.','frozen_update':'delta_final=product(later residual lambdas)*W_O_head*delta_head; allnative attention/MLP updates including currentMLP held fixed. Read out nativefinalstate+delta_final with exactfinalRMS/unembedding/30tanh. Also fixednativeRMS andno-logitcap variants. This is a counterfactual skip-path approximation, not thelive model.','transfer':'Predict livecapital response from measuredlivecopy response times nativeRMS_copy/RMS_capital or analytic skipresponse_capital/skipresponse_copy. Analyticratio is diagnostic; report any zero denominator. No new taskgainfit.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);actual=0;nativeerr=0.;readouterr=0.;skiperr=0.;records=[];saved={};eps=torch.finfo(torch.float32).eps
 for panel in ['old','new']:
  data=torch.load(paths[panel+'_panel'],weights_only=False);ex=data['examples'];screen=json.loads(paths[panel+'_screen'].read_text());live=json.loads(paths[panel+'_live'].read_text());lookup={(r['id'],r['order'],r['layer'],r['head'],r['mode']):r for r in live['records']};idx={(e['id'],e['order']):i for i,e in enumerate(ex)};buckets=defaultdict(list)
  for i,e in enumerate(ex):buckets[len(e['ids'])].append(i)
  finals=torch.empty(len(ex),1152);stash={}
  def hook(mod,args,out):stash['final']=out[0][:,-1].detach().cpu()
  handle=model.blocks[-1].register_forward_hook(hook)
  try:
   for length,ii in sorted(buckets.items()):
    for st in range(0,len(ii),12):
     ix=ii[st:st+12];y=final_logits(model,torch.tensor([ex[i]['ids'] for i in ix],device='cuda'));actual+=len(ix);finals[ix]=stash['final'];nativeerr=max(nativeerr,float((y.double().cpu()-torch.stack([data['native'][i] for i in ix])).abs().max()))
  finally:handle.remove()
  saved[panel]=finals;U=model.lm_head.weight.double().cpu();bias=None if model.lm_head.bias is None else model.lm_head.bias.double().cpu()
  for i,e in enumerate(ex):
   b=finals[i].double();r=(b.square().mean()+eps).sqrt();j=idx[e['id'],1-e['order']];pair=screen['config']['pairs'][e['pair']];outputids={typ:[continuation_id(tok,e['text'],' '+v[part]) for v in pair] for typ,part in [('country',0),('capital',1)]};assert all(v is not None for ii in outputids.values() for v in ii)
   for l,h in TARGETS:
    key=(e['id'],e['order'],l,h);p=lookup[key+('common_positive',)];m=lookup[key+('common_negative',)];amp=p['routing_coefficient'];assert amp>0 and abs(amp+m['routing_coefficient'])<1e-10;source=e['source_positions'][0];db=(data['bus'][j][source,h]-data['bus'][i][source,h]).double();W=model.blocks[l].attn.c_proj.weight[:,h*128:(h+1)*128].double().cpu();write=W@(amp*db);factor=1.
    for block in model.blocks[l+1:]:factor*=float(block.lambdas[0])
    delta=write*factor;iterative=write.clone()
    for block in model.blocks[l+1:]:iterative*=float(block.lambdas[0])
    skiperr=max(skiperr,float((iterative-delta).abs().max()))
    for typ,ids in outputids.items():
     w=U[ids];bb=torch.zeros(2,dtype=torch.double) if bias is None else bias[ids];baseline=30*torch.tanh((w@(b/r)+bb)/30);readouterr=max(readouterr,float((baseline-data['native'][i][ids]).abs().max()));response={}
     for variant in ['exact','fixed_rms','no_cap']:
      margins=[]
      for sign in [1,-1]:
       z=b+sign*delta;den=r if variant=='fixed_rms' else (z.square().mean()+eps).sqrt();logits=w@(z/den)+bb
       if variant!='no_cap':logits=30*torch.tanh(logits/30)
       margins.append(float(logits[0]-logits[1]))
      response[variant]=(margins[0]-margins[1])/(2*amp)
     records.append({k:e[k] for k in ['pair','format','split','family','order','id']}|{'panel':panel,'layer':l,'head':h,'output':typ,'native_final_RMS':float(r),'amplitude':amp,'skip_factor':factor,'skip_delta_norm':float(delta.norm()),'live_response':(p['edited_'+typ+'_margin']-m['edited_'+typ+'_margin'])/(2*amp),'skip_responses':response})
  print('DONE',panel,len(ex),flush=True);del U,data
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_abs':nativeerr,'exact_readout_native_max_abs':readouterr,'skip_recurrence_max_abs':skiperr,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['live_response']) and all(np.isfinite(v) for v in r['skip_responses'].values()) for r in records)};assert audits['hashes_unchanged'] and max(nativeerr,readouterr)<1e-4 and skiperr<1e-8 and audits['finite'];torch.save(saved,ART/f'{S}_final_states.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
