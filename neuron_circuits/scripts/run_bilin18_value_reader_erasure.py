"""H867 full factorial of residual, persistent value bus, and embedding anchor."""
import json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_value_reader_erasure'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_task_state_location_captures.pt','location':ART/'bilin18_task_state_location.json','examples':ART/'bilin18_candidate_task_screen_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','prior':ART/'bilin18_value_slice_erasure.json'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H867','hashes':hashes,'design':'All16 downstream layers2..17 andall9 valuebus slices, each128dimensions. Atboundary1 preserve nativeR/E. Apply midpoint slice only to selectedreader layer, restore nativebus for all otherlayers while allowing residual effects to propagate. All76pairs/all5taskfamilies. Native,fullbus andall9 global-slice controls replayH866.','interpretation':'Head-index slice affects all downstream readers at that head index, not one layer/head. Bus tokenlocal content is distinct from contextual routing. Single/leave-one-out erasures expose conditional contributions and nonadditivity, not unique attribution. Midpoints can be off-manifold. Earlier H739staticbus and H778routing/value evidence are prior context.','scope':'Post-H843 exploratory route diagnosis, frozen task rows. All architecture-specific persistent inputs included. No feature selection or reserved test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['captures'],weights_only=False);caps,native=data['captures'],data['native'];ex=torch.load(paths['examples'],weights_only=False)['examples'];loc=json.loads(paths['location'].read_text());jobs=[r for r in loc['records'] if r['mode']=='complete'];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];nativeerr=0.;donorerr=0.;actual=0;paired_logits={};bus_passthrough=0.;local_application_counts=set()
 for l in [1]:
  lengths=sorted({len(ex[r['i']]['ids']) for r in jobs if r['layer']==l})
  for length in lengths:
   jj=[r for r in jobs if r['layer']==l and len(ex[r['i']]['ids'])==length]
   for mode,head,reader in [('native',-1,-1),('all',-1,-1)]+[('single',h,-1) for h in range(9)]+[('reader',h,layer) for layer in range(2,18) for h in range(9)]:
    for st in range(0,len(jj),24):
     batch=jj[st:st+24]
     x=torch.stack([caps[r['i']][l,'sequence'] for r in batch]).cuda();x0=torch.stack([caps[r['i']]['x0'] for r in batch]).cuda();original_bus=torch.stack([caps[r['i']][l,'bus'] for r in batch]).cuda();donor_bus=torch.stack([caps[r['j']][l,'bus'] for r in batch]).cuda();assert original_bus.shape[-2:]==(9,128)
     selected=torch.zeros(9,dtype=torch.bool,device='cuda')
     if mode=='all':selected[:]=True
     if mode in ['single','reader']:selected[head]=True
     mid=((original_bus.double()+donor_bus.double())*.5).float();bus=torch.where(selected[None,None,:,None],mid,original_bus);unchanged=torch.where(selected[None,None,:,None],torch.zeros_like(bus),bus-original_bus);assert float(unchanged.abs().max())==0
     edits=(bus-original_bus).flatten(1).norm(dim=1);delta=(donor_bus-original_bus).flatten(1).norm(dim=1)
     edited_bus=bus;applied=0
     for layer,block in enumerate(model.blocks[l+1:],start=l+1):
      apply=(mode!='reader' or layer==reader);provided=edited_bus if apply else original_bus;applied+=int(mode=='reader' and apply);x,returned=block(x,provided,x0);bus_passthrough=max(bus_passthrough,float((returned-provided).abs().max()))
     if mode=='reader':local_application_counts.add(applied);assert applied==1
     y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30);actual+=len(batch)
     for k,r in enumerate(batch):
      i,j=r['i'],r['j'];base=measure(native[i],ex[i]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;lp=native[i].log_softmax(0);records.append({key:r[key] for key in ['layer','family','split','id','recipient','donor','i','j','native_answer_gap']}|{'mode':mode,'head':head,'reader_layer':reader,'edit_norm':float(edits[k]),'bus_difference_norm':float(delta[k]),'native_margin':base,'erased_margin':value,'native_top1_correct':measure(native[i],ex[i]['groups'])['top1'][r['recipient']],'erased_top1_correct':measure(y[k],ex[i]['groups'])['top1'][r['recipient']],'donor_directed_delta':sgn*(value-base),'KL':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())})
      if mode=='native':nativeerr=max(nativeerr,float((y[k].double().cpu()-native[i]).abs().max()))
   print('DONE',l,length,flush=True)
 prior=json.loads(paths['prior'].read_text());old={(r['id'],r['recipient'],r['mode'],r['head']):r['erased_margin'] for r in prior['records']};replay=max(abs(r['erased_margin']-old[r['id'],r['recipient'],r['mode'],r['head']]) for r in records if r['mode']!='reader')
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_logits':nativeerr,'prior_global_control_margin_error':replay,'bus_passthrough_max_abs':bus_passthrough,'reader_application_counts':sorted(local_application_counts),'partial_suffix_forwards':actual,'whole_model_forwards':0,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) for r in records)};assert audits['hashes_unchanged'] and max(nativeerr,replay)<1e-4 and audits['finite'] and bus_passthrough==0 and local_application_counts=={1} and actual==152*155;(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'audits':audits}),flush=True)
if __name__=='__main__':main()
