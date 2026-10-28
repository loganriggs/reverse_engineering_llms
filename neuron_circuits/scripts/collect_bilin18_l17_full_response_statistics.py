"""H285 fixed-feature intact KL plus exact quadratic response loss."""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from bilin18_residual_cross_program import intact_forward,change
from run_bilin18_l17_fixed_readout import read_features
from bilin18_readout_feature_families import features as outside_features
from run_bilin18_l17_kl_intercept import distillation_kl
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_response_regularized'
def response_loss(w,g,r,e):
    return ((w*(g@w)).sum()-2*(w*r).sum()+e)/e
@torch.no_grad()
def stats(df,target):return df.double().T@df.double(),df.double().T@target.double(),target.double().square().sum()
def main():
 global STEM
 if len(sys.argv)>1:STEM=ART/sys.argv[1]
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 for v in model.lm_head.parameters():v.requires_grad_(False)
 block=model.blocks[17];L,R,D=[m.weight.detach() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');pr=load_program(ART/'bilin18_l17_residual_cross_actual_refined_program.npz','cuda');f=load_program(ART/'bilin18_l17_fixed_readout_frozen.npz','cuda');outside=load_program(ART/'bilin18_l17_feature_outside_frozen.npz','cuda');cut=768*4
 def logits(s):return 30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)
 with torch.no_grad():
  pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);write=compiled+intact_forward(x,pr);base=pre+write+block.mlp.Down_bias;feat=torch.cat([read_features(x,pr,f['feature_mean'],f['feature_std']),outside_features(x,pr,outside)],1);energy=(native[:cut]-write[:cut]).square().sum(1).mean();teacher=torch.cat([logits(pre[i:i+16]+native[i:i+16]+block.mlp.Down_bias).double().log_softmax(-1) for i in range(0,len(x),16)]);groups={};rng=torch.Generator(device='cuda').manual_seed(cfg['seed'])
  for h in range(9):
   source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;actual=heads[:,h];angle=angle_matched_payload(pre,actual,source,torch.randn(len(x),128,device='cuda',generator=rng))
   for label,payload in [('actual',actual),('angle',angle)]:
    z=F.rms_norm(pre-payload,(1152,));df=z-x;target=B(z,z,L,R,D)-native-(forward_normalized(z,p)-compiled)-change(x,z,pr)
    for split,sl in [('train',slice(0,cut)),('validation',slice(cut,None))]:
     key=f'{split}_{label}_'+('heads0_5' if h<6 else 'heads6_8');values=stats(df[sl],target[sl]);groups[key]=[a+b for a,b in zip(groups[key],values)] if key in groups else list(values)
   print('response stats head',h,flush=True)
  train=[sum(groups['train_'+label+'_'+group][k] for label in ['actual','angle'] for group in ['heads0_5','heads6_8']) for k in range(3)];train=[v.float() for v in train]
 np.savez_compressed(STEM.with_name(STEM.name+'_statistics.npz'),state_mean=x[:cut].double().mean(0).cpu().numpy(),state_cov=((x[:cut].double()-x[:cut].double().mean(0)).T@(x[:cut].double()-x[:cut].double().mean(0))/cut).cpu().numpy(),**{k+'_'+name:v.cpu().numpy() for k,vs in groups.items() for name,v in zip(['gram','rhs','energy'],vs)});print('FULL STATISTICS SAVED',flush=True)
if __name__=='__main__':main()
