"""H441 baseline-only calibration coverage/shrinkage selection."""
import json,hashlib
import numpy as np
import torch
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_calibration_coverage'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 calcfg=json.loads((ROOT/'configs/bilin18_l17_baseline_context_confirmation.json').read_text());assert hashlib.sha256((ROOT/calcfg['corpus']).read_bytes()).hexdigest()==calcfg['corpus_sha256'];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,_,_=collect_sources(model,np.load(ROOT/calcfg['corpus']),calcfg);x=pre.double()/(pre.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();maps=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');activity=torch.einsum('roi,ni->nro',maps,x);order=np.random.default_rng(cfg['split_seed']).permutation(512);train=order[:384];val=order[384:];weight=torch.einsum('roi,rpi->op',maps,maps);av=activity[val].reshape(-1,1152);den=av.square().sum();rows=[];bases=[];selected=[]
 for n in cfg['training_counts']:
  ids=train[:n];aa=activity[ids].reshape(-1,1152);gram=aa.T@aa/n;isotropic=x[ids].square().mean();local=[]
  for mix in cfg['mixing_weights']:
   g=mix*gram+(1-mix)*isotropic*weight;_,b=torch.linalg.eigh(g);b=b[:,-cfg['rank']:].flip(-1);valerror=float((av.square().sum()-(av@b).square().sum())/den);trainerror=float((aa.square().sum()-(aa@b).square().sum())/aa.square().sum());row={'training_count':n,'activity_weight':mix,'validation_activity_error_fraction':valerror,'training_activity_error_fraction':trainerror};rows.append(row);bases.append(b.cpu().numpy());local.append(len(rows)-1);print(row,flush=True)
  selected.append(min(local,key=lambda i:(rows[i]['validation_activity_error_fraction'],i)))
 best=min(range(len(rows)),key=lambda i:(rows[i]['validation_activity_error_fraction'],i));np.savez_compressed(STEM.with_name(STEM.name+'_frozen_bases.npz'),bases=np.stack(bases),training_ids=train,validation_ids=val,calibration_pre=pre.cpu().numpy());selection={'config':cfg,'candidates':rows,'selected_by_count':selected,'overall_selected':best,'selection_rule':'min reserved baseline activity error; exact ties use earliest candidate; no H438 outcomes used','unique_completed_documents':12480};STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('SELECTION FROZEN',selected,best,flush=True)
 # Only after selection: test unweighted activity on the separate H438 panel.
 test=np.load(ART/'bilin18_l17_operator_transfer_confirmation_rows.npz');tp=torch.tensor(test['pre'],device='cuda').double();tx=tp/(tp.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();ta=torch.einsum('roi,ni->nro',maps,tx).reshape(-1,1152);selection['transfer_activity_error_fractions']=[float((ta.square().sum()-(ta@torch.tensor(b,device='cuda')).square().sum())/ta.square().sum()) for b in bases];STEM.with_suffix('.json').write_text(json.dumps(selection,indent=2)+'\n');print('FINISHED',flush=True)
if __name__=='__main__':main()
