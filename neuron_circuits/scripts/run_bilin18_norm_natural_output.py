"""H858 whole-output natural-text check of frozen single-state calibration."""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_early_local_geometry import sha
S='bilin18_norm_natural_output'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'predictions':ART/'bilin18_single_state_norm_calibration_predictions.pt','natural':ART/'bilin18_relay_natural_data_captures.pt','source':ART/'bilin18_fresh_early_geometry_panel_captures.pt','taskdata':ART/'bilin18_relay_fresh_distance_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H858','hashes':hashes,'design':'First64 H851 naturaldevelopment occurrences, frozen before whole-output outcomes. All40 H857 predictions/variants. Truncate native128token window after selected causal position, replace full rawpreMLP4 state there, compare nexttoken distributions.','scope':'Whole representation replacement, not donor contrast intervention. Reused development data; no newtraining/test. KL(native||replacement) fidelity distinct from nexttokenCE. Labels available only when selected position<127; all64 retained forKL.','controls':'Native reconstruction identity and capture replay; exact saved rawstate. Paired document comparisons, no model/seed selection.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');pred=torch.load(paths['predictions'],weights_only=False)['predictions'];data=torch.load(paths['natural'],weights_only=False);source=torch.load(paths['source'],weights_only=False);N=len(torch.load(paths['taskdata'],weights_only=False)['examples']);ids=source['all_token_ids'][data['validation_documents'][:64]].clone();pos=[7+8*k for k in data['validation_positions'][:64]];del source;model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];actual=0;captureerr=0.;ident=0.
 for position in sorted(set(pos)):
  inds=[i for i,p in enumerate(pos) if p==position]
  for st in range(0,len(inds),8):
   batch=inds[st:st+8];tokens=ids[batch,:position+1].cuda();x0=F.rms_norm(model.embedding(tokens),(1152,));x=x0;bus=None
   for l,block in enumerate(model.blocks):
    raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att
    if l==4:captureerr=max(captureerr,float((pre[:,-1].cpu()-data['captures']['validation',4][batch]).abs().max()))
    x=pre+block.mlp(F.rms_norm(pre,(1152,)))
   native=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30);actual+=len(batch);positions=torch.full((len(batch),),position,device='cuda',dtype=torch.long);own=data['captures']['validation',4][batch].cuda();yy=run(model,tokens,own,positions);actual+=len(batch);ident=max(ident,float((yy-native).abs().max()));lp=native.double().log_softmax(-1)
   for (name,variant),prediction in pred.items():
    y=run(model,tokens,prediction[[N+i for i in batch]].cuda(),positions);actual+=len(batch);lq=y.double().log_softmax(-1);kl=(lp.exp()*(lp-lq)).sum(-1);tv=(lp.exp()-lq.exp()).abs().sum(-1)/2
    for k,i in enumerate(batch):
     label=int(ids[i,position+1]) if position<127 else None;records.append({'document_index':i,'source_document_index':data['validation_documents'][i],'position':position,'model':name,'variant':variant,'KL':float(kl[k]),'TV':float(tv[k]),'next_token_CE_change':float(lp[k,label]-lq[k,label]) if label is not None else None})
  print('DONE',position,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_capture_max_abs':captureerr,'identity_logits_max_abs':ident,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) for r in records)};print(audits,flush=True);(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');assert audits['hashes_unchanged'] and captureerr<.01 and ident<1e-4 and audits['finite']
if __name__=='__main__':main()
