"""H432 post-selected response-origin/output-recipient cross."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
from run_bilin18_qkov_causal_edges import TOKENIZER,AutoTokenizer
STEM=ART/'bilin18_l17_output_metric_cross'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 old=np.load(ART/'bilin18_l17_tail_source_recipient_cross_rows.npz');states=np.load(ART/'bilin18_l17_recipient_tail_diagnostic_current_states.npz');sites=cfg['sites'];pre=torch.tensor(states['pre'][sites],device='cuda');st=torch.tensor(states['state'][sites],device='cuda');base=torch.tensor(states['base'][sites],device='cuda');sources=torch.tensor(old['sources'],device='cuda');model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();x=F.rms_norm(pre,(1152,));m0=((x@L.T)*(x@R.T))@D.T;responses=[]
 for h in sources:
  xx=F.rms_norm(pre-h.float(),(1152,));responses.append(((xx@L.T)*(xx@R.T))@D.T-m0)
 responses=torch.stack(responses);norms=responses.double().norm(dim=-1);common=norms.flatten().median();vals=np.empty((2,48,17,17,2));token_rows=[];tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 for dst in range(17):
  ss=st[dst:dst+1];bb=base[dst:dst+1];th=torch.tanh(bb.double()@W.T/30);cap=1-th.square();lp=(30*th).log_softmax(-1);p=lp.exp();acc=torch.zeros(W.shape[0],device='cuda',dtype=torch.float64);targetacc=acc.clone()
  for ni in range(2):
   rr=responses if ni==0 else responses.double()*(common/norms)[:,:,None];flat=rr.reshape(-1,1152)
   for off in range(0,len(flat),32):
    v=flat[off:off+32].double();dt=(native_drms(ss.double(),v)@W.T)*cap;dt-= (p*dt).sum(-1,keepdim=True);fe=.5*(p*dt.square()).sum(-1);target=F.rms_norm(ss+v.float(),(1152,));lt=(30*torch.tanh(target.double()@W.T/30)).log_softmax(-1);terms=lt.exp()*(lt-lp);kl=terms.sum(-1);arr=torch.stack([fe,kl],-1).cpu().numpy();vals[ni,:,:,dst].reshape(-1,2)[off:off+len(v)]=arr
    if ni==0:
     ids=torch.arange(off,off+len(v),device='cuda');sel=(ids%17==dst)
     if sel.any():acc+=terms[sel].sum(0);targetacc+=lt[sel].exp().sum(0)
  acc/=48;targetacc/=48;top=acc.abs().topk(12).indices;token_rows.append({'destination':dst,'site':sites[dst],'matched_origin_mean_KL':float(acc.sum()),'largest_absolute_signed_token_contributions':[{'id':int(t),'text':tok.decode([int(t)]),'signed_KL_contribution':float(acc[t]),'baseline_probability':float(p[0,t]),'mean_target_probability':float(targetacc[t])} for t in top]})
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),values=vals[:,:,:,:dst+1]);print('destination',dst,'complete',flush=True)
 assert np.isfinite(vals).all();ref=old['values'][:,0,0,1,:,1:];diagonal=np.stack([vals[0,:,i,i] for i in range(17)],1);err=float(np.max(abs(diagonal-ref)/np.maximum(abs(ref),1e-12)));np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),values=vals,responses=responses.cpu().numpy(),response_norms=norms.cpu().numpy());means=vals.mean(1);cells=np.stack([np.stack([means[:,0,0],means[:,0,1:].mean(1)],1),np.stack([means[:,1:,0].mean(1),means[:,1:,1:].mean((1,2))],1)],1);j={'experiment':'H432','config':cfg,'common_response_norm':float(common),'cell_axes':['response_norm_control','origin_tail_or_controls','destination_tail_or_controls','Fisher_or_KL'],'cells':cells.tolist(),'native_diagonal_relative_replay_max':err,'native_replay_gate':err<cfg['replay_tolerance'],'token_diagnostics':token_rows,'unique_completed_documents':11456};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',err,cells,flush=True)
if __name__=='__main__':main()
