"""H865 full factorial of residual, persistent value bus, and embedding anchor."""
import json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_diverse_route_erasure'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_task_state_location_captures.pt','location':ART/'bilin18_task_state_location.json','examples':ART/'bilin18_candidate_task_screen_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H865','hashes':hashes,'design':'All8 subsets of pair-midpoint residual sequence R, persistent layer0 value bus V, original normalized embedding anchor E, at boundary1.76 equal-length H843 task pairs across5 competent families, both recipients. All-state midpoint must give equal logits for paired contexts; native state replay audited.','interpretation':'Pair-midpoint erasure of route contrasts, not donor replacement. Report native pair-margin distinction and answer correctness. Routes can interact or redundantly carry information; no unique mediation attribution. Equal token length, not semantic alignment. Midpoint states can be off native manifold.','scope':'Post-H843 exploratory route diagnosis, frozen task rows. All architecture-specific persistent inputs included. No feature selection or reserved test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['captures'],weights_only=False);caps,native=data['captures'],data['native'];ex=torch.load(paths['examples'],weights_only=False)['examples'];loc=json.loads(paths['location'].read_text());jobs=[r for r in loc['records'] if r['mode']=='complete'];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];nativeerr=0.;donorerr=0.;actual=0;paired_logits={}
 for l in [1]:
  lengths=sorted({len(ex[r['i']]['ids']) for r in jobs if r['layer']==l})
  for length in lengths:
   jj=[r for r in jobs if r['layer']==l and len(ex[r['i']]['ids'])==length]
   for mask in range(8):
    for st in range(0,len(jj),24):
     batch=jj[st:st+24]
     def midpoint(field,bit):
      def get(i):return caps[i]['x0'] if field=='x0' else caps[i][l,field]
      return torch.stack([(get(r['i']).double()+get(r['j']).double())*.5 if mask&bit else get(r['i']).double() for r in batch]).float().cuda()
     x=midpoint('sequence',1);bus=midpoint('bus',2);x0=midpoint('x0',4)
     for block in model.blocks[l+1:]:x,bus=block(x,bus,x0)
     y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30);actual+=len(batch)
     for k,r in enumerate(batch):
      i,j=r['i'],r['j'];base=measure(native[i],ex[i]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;lp=native[i].log_softmax(0);records.append({key:r[key] for key in ['layer','family','split','id','recipient','donor','i','j','native_answer_gap']}|{'mask':mask,'native_margin':base,'erased_margin':value,'native_top1_correct':measure(native[i],ex[i]['groups'])['top1'][r['recipient']],'erased_top1_correct':measure(y[k],ex[i]['groups'])['top1'][r['recipient']],'donor_directed_delta':sgn*(value-base),'KL':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())})
      if mask==0:nativeerr=max(nativeerr,float((y[k].double().cpu()-native[i]).abs().max()))
      if mask==7:
       key=(l,r['id']);previous=paired_logits.get(key)
       if previous is not None:donorerr=max(donorerr,float((y[k].double().cpu()-previous).abs().max()))
       paired_logits[key]=y[k].double().cpu()
   print('DONE',l,length,flush=True)
 summary=[]
 for l in [1]:
  for fam in sorted({r['family'] for r in records}):
   for mask in range(8):
    rr=[r for r in records if (r['layer'],r['family'],r['mask'])==(l,fam,mask)];summary.append({'layer':l,'family':fam,'mask':mask,'gap_fraction':sum(r['donor_directed_delta'] for r in rr)/sum(r['native_answer_gap'] for r in rr),'pairs':len(rr)//2})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_replay_max_logits':nativeerr,'all_midpoint_paired_logits_error':donorerr,'partial_suffix_forwards':actual,'whole_model_forwards':0,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) for r in records)};assert audits['hashes_unchanged'] and max(nativeerr,donorerr)<1e-4 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits}),flush=True)
if __name__=='__main__':main()
