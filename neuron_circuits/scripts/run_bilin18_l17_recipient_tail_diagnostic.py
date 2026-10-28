"""H430 descriptive recipient-tail tracing; no exclusions or confirmatory selection."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_qkov_causal_edges import TOKENIZER,AutoTokenizer
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_recipient_tail_diagnostic'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);results={};saved={}
 for label,conf in [('current','bilin18_l17_centered_source_reflection'),('previous','bilin18_l17_normalization_curvature_factorial')]:
  cfg=json.loads((ROOT/'configs'/f'{conf}.json').read_text());path=ROOT/cfg['corpus'];assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['corpus_sha256'];tokens=np.load(path);pre,heads,_=collect_sources(model,tokens,cfg);x=F.rms_norm(pre,(1152,));state=pre+((x@L.T)*(x@R.T))@D.T+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));stats=np.empty((len(pre),5));stats[:,0]=pre.double().square().mean(-1).sqrt().cpu();stats[:,1]=state.double().square().mean(-1).sqrt().cpu();stats[:,2]=heads[:,6].double().norm(dim=-1).cpu()
  for off in range(0,len(pre),16):
   lp=(30*torch.tanh(base[off:off+16].double()@W.T/30)).log_softmax(-1);p=lp.exp();stats[off:off+16,3]=(-(p*lp).sum(-1)).cpu();stats[off:off+16,4]=p.max(-1).values.cpu()
  if label=='current':
   c=np.load(ART/'bilin18_l17_centered_source_reflection_rows.npz')['coefficients'];quartic=c[:,:,:,2,0].mean((1,2));quadratic=c[:,:,:,0,0].mean((1,2))
  else:
   c=np.load(ART/'bilin18_l17_normalization_curvature_factorial_bilinear_polynomial_coefficients.npz')['coefficients'];quartic=np.stack([c[2,ix,0].mean(0) for ix in [np.arange(1,13),np.arange(13,25)]]);quadratic=np.stack([c[0,ix,0].mean(0) for ix in [np.arange(1,13),np.arange(13,25)]])
  selected=np.argsort(quartic.mean(0))[-8:][::-1];cases=[]
  for si in selected:
   doc,pi=divmod(int(si),4);pos=cfg['positions'][pi];cases.append({'site':int(si),'document':doc,'position':pos,'context':tok.decode(tokens[doc,max(0,pos-24):pos+1].tolist()),'stats':stats[si].tolist(),'stats_percentiles':(stats<=stats[si]).mean(0).tolist(),'quartic_energy':quartic[:,si].tolist(),'quartic_total_share':(quartic[:,si]/quartic.sum(1)).tolist()})
  results[label]={'stats_names':['input_rms','output_rms','actual_head6_norm','baseline_entropy','baseline_max_probability'],'stats_median':np.median(stats,axis=0).tolist(),'stats_quantiles':np.quantile(stats,[.01,.5,.99],axis=0).tolist(),'max_document_quartic_share':(quartic.reshape(2,-1,4).sum(2).max(1)/quartic.sum(1)).tolist(),'max_site_quartic_share':(quartic.max(1)/quartic.sum(1)).tolist(),'top_sites':cases}
  saved.update({label+'_stats':stats,label+'_quartic':quartic,label+'_quadratic':quadratic})
  if label=='current':
   # Save receiving states so selected-site controlled followups need not rerun the corpus.
   np.savez_compressed(STEM.with_name(STEM.name+'_current_states.npz'),pre=pre.cpu().numpy(),state=state.cpu().numpy(),base=base.cpu().numpy(),head6=heads[:,6].cpu().numpy())
   # Source-independent bound on input RMS derivative gain (1/input_rms).
   si=199*4+1;results[label]['selected_document199_site63']=next(v for v in cases if v['site']==si)
  print(label,'complete',results[label]['max_site_quartic_share'],flush=True)
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H430','scope':'Post-selection descriptive diagnostic. Current reflection-averaged versus previous unreflected sources differ; panel comparison is not a controlled causal effect. No exclusions, no fresh documents.','panels':results,'unique_completed_documents':11456},indent=2)+'\n');print('FINISHED',flush=True)
if __name__=='__main__':main()
