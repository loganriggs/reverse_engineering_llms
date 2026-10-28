"""H283 validation-only selection of frozen correction mixtures and shrinkage controls."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from bilin18_residual_cross_program import intact_forward
from bilin18_readout_feature_families import features as family_features
from run_bilin18_l17_fixed_readout import read_features
from run_bilin18_l17_kl_intercept import distillation_kl
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs/bilin18_l17_feature_mixture.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();block=model.blocks[17];L,R,D=[m.weight for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');pr=load_program(ART/'bilin18_l17_residual_cross_actual_refined_program.npz','cuda')
 corpus=np.load(ROOT/cfg['discovery'],mmap_mode='r')[768:1024];pre,_,_=collect_sources(model,corpus,{**cfg,'rows':256,'start':0});x=F.rms_norm(pre,(1152,));base=pre+forward_normalized(x,p)+intact_forward(x,pr)+block.mlp.Down_bias;native=pre+B(x,x,L,R,D)+block.mlp.Down_bias;cs=[]
 for n,stem in cfg['stems'].items():
  f=load_program(ART/f'{stem}_frozen.npz','cuda');feat=read_features(x,pr,f['feature_mean'],f['feature_std']) if n=='linear' else family_features(x,pr,f);cs.append(feat@f['kl'])
 corrections=torch.stack(cs);grid=[(i/10,j/10,k/10) for i in range(11) for j in range(11-i) for k in range(11-i-j)];losses=torch.zeros(len(grid),device='cuda',dtype=torch.float64)
 def logits(s):return 30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)
 for start in range(0,len(x),16):
  sl=slice(start,start+16);lp=logits(native[sl]).double().log_softmax(-1)
  for k,w in enumerate(grid):losses[k]+=distillation_kl(logits(base[sl]+sum(w[i]*corrections[i,sl] for i in range(3))),lp).sum()
  if start%128==0:print('validation sites',start,flush=True)
 losses=(losses/len(x)).cpu().numpy();winner=int(np.argmin(losses));pure=[i for i,w in enumerate(grid) if sum(v>0 for v in w)<=1];shrink=min(pure,key=lambda i:losses[i]);single=[i for i,w in enumerate(grid) if sum(v==1 for v in w)==1];best=min(single,key=lambda i:losses[i]);result={'config':cfg,'grid':grid,'validation_kl':losses.tolist(),'mixture_weights':grid[winner],'shrink_weights':grid[shrink],'single_weights':grid[best],'mixture_kl':float(losses[winner]),'shrink_kl':float(losses[shrink]),'single_kl':float(losses[best])};(ART/'bilin18_l17_feature_mixture_selection.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['config','grid','validation_kl']},flush=True)
if __name__=='__main__':main()
