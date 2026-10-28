"""H285 fixed-feature intact KL plus exact quadratic response loss."""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from bilin18_residual_cross_program import intact_forward,change
from run_bilin18_l17_fixed_readout import read_features
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
 block=model.blocks[17];L,R,D=[m.weight.detach() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');pr=load_program(ART/'bilin18_l17_residual_cross_actual_refined_program.npz','cuda');f=load_program(ART/'bilin18_l17_fixed_readout_frozen.npz','cuda');cut=768*4
 def logits(s):return 30*torch.tanh(model.lm_head(F.rms_norm(s,(1152,)))/30)
 with torch.no_grad():
  pre,heads,_=collect_sources(model,np.load(ROOT/cfg['discovery'],mmap_mode='r'),{**cfg,'rows':1024});x=F.rms_norm(pre,(1152,));native=B(x,x,L,R,D);compiled=forward_normalized(x,p);write=compiled+intact_forward(x,pr);base=pre+write+block.mlp.Down_bias;feat=read_features(x,pr,f['feature_mean'],f['feature_std']);energy=(native[:cut]-write[:cut]).square().sum(1).mean();teacher=torch.cat([logits(pre[i:i+16]+native[i:i+16]+block.mlp.Down_bias).double().log_softmax(-1) for i in range(0,len(x),16)]);groups={};rng=torch.Generator(device='cuda').manual_seed(cfg['seed'])
  for h in range(9):
   source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;actual=heads[:,h];angle=angle_matched_payload(pre,actual,source,torch.randn(len(x),128,device='cuda',generator=rng))
   for label,payload in [('actual',actual),('angle',angle)]:
    z=F.rms_norm(pre-payload,(1152,));df=((z-x)@torch.cat([pr['core'],pr['partner']],1))/f['feature_std'];target=B(z,z,L,R,D)-native-(forward_normalized(z,p)-compiled)-change(x,z,pr)
    for split,sl in [('train',slice(0,cut)),('validation',slice(cut,None))]:
     key=f'{split}_{label}_'+('fit' if h in cfg['fit_heads'] else 'heldout');values=stats(df[sl],target[sl]);groups[key]=[a+b for a,b in zip(groups[key],values)] if key in groups else list(values)
   print('response stats head',h,flush=True)
  train=[sum(groups['train_'+label+'_fit'][k] for label in ['actual','angle']) for k in range(3)];train=[v.float() for v in train]
 @torch.no_grad()
 def score(w):
  kl={}
  for split,start,end in [('train',0,cut),('validation',cut,len(x))]:
   total=0.
   for i in range(start,end,16):total+=distillation_kl(logits(base[i:i+16]+feat[i:i+16]@w),teacher[i:i+16]).sum()
   kl[split]=float(total/(end-start))
  return {'kl':kl,'response_relative_error':{k:float(response_loss(w.double(),*v)) for k,v in groups.items()}}
 zero=torch.zeros_like(f['kl']);baseline=score(zero);old=score(f['kl']);curves={};weights={'base':zero,'h280':f['kl']};records={}
 for strength in cfg['strengths']:
  for rate in cfg['rates']:
   key=f'lambda{strength}_lr{rate}';w=torch.nn.Parameter(f['kl'].clone());opt=torch.optim.Adam([w],lr=rate);gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+1);curves[key]=[]
   for step in range(cfg['steps']+1):
    if step in cfg['checkpoints']:
     rec=score(w);name=f'{key}_s{step}';curves[key].append({'step':step,**rec});records[name]=rec;weights[name]=w.detach().clone();print(name,rec['kl']['validation'],{k:v for k,v in rec['response_relative_error'].items() if k.startswith('validation')},flush=True)
    if step==cfg['steps']:break
    ix=torch.randint(cut,(32,),generator=gen,device='cuda');corr=feat[ix]@w;loss=distillation_kl(logits(base[ix]+corr),teacher[ix]).mean()+.1*corr.square().sum(1).mean()/energy+strength*baseline['kl']['train']*response_loss(w,*train);opt.zero_grad();loss.backward();opt.step()
 # Validation selection ignores held-out heads completely.
 def resp(rec):return max(rec['response_relative_error']['validation_'+label+'_fit'] for label in ['actual','angle'])
 eligible=[n for n,v in records.items() if v['kl']['validation']<=1.02*old['kl']['validation'] and resp(v)<=1.0]
 winner=min(eligible,key=lambda n:records[n]['kl']['validation']) if eligible else min(records,key=lambda n:records[n]['kl']['validation']/old['kl']['validation']+resp(records[n]))
 weights['selected']=weights[winner];np.savez_compressed(STEM.with_name(STEM.name+'_frozen.npz'),feature_mean=f['feature_mean'].cpu().numpy(),feature_std=f['feature_std'].cpu().numpy(),**{n:w.cpu().numpy() for n,w in weights.items()});STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'config':cfg,'baseline':baseline,'h280':old,'curves':curves,'records':records,'eligible':eligible,'winner':winner,'selection_mode':'feasible' if eligible else 'fallback_joint','selected':records[winner]},indent=2));print('SELECTED',winner,'eligible',len(eligible),flush=True)
if __name__=='__main__':main()
