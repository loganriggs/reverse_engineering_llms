"""H830 exact RMS/bilinear derivative factor crossing on saved contexts."""
import json,time,itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_early_local_geometry import sha
S='bilin18_mlp_state_input_cross'
def cos(a,b):return float(a@b/(a.norm()*b.norm()).clamp_min(1e-30))
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'trace':ART/'bilin18_fixed_direction_depth_captures.pt','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};d=torch.load(paths['trace'],weights_only=False);cfg={'experiment':'H830','hashes':hashes,'design':'All13 H827 groups and block/random directions. At first downstream MLP cross each of4 natural raw baselines with each of4 captured incoming central derivatives. No averaged/off-data baseline.','derivative':'u=b/r; q=v/r-b*mean(b*v)/r^3; r=sqrt(mean(b^2)+float32eps). Output D[(Lu)*(Rq)+(Lq)*(Ru)]. Double precision math with fixed model weights.','comparisons':'Own diagonal; hold each baseline fixed across4 inputs; hold each input fixed across4 baselines. Average six cosines and all4 reference choices without selecting best. Additional normalizer-vs-bilinear-state decomposition at fixed raw input direction.','scope':'Analytic local module sensitivity, not whole-model causal mediation or semantic feature discovery. Incoming v is captured finite central derivative; verify natural replay and finite differences.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda');records=[];replays=[];rmsreplay=[];fd=[];captures={};eps=torch.finfo(torch.float32).eps
 for gi,g in enumerate(d['groups']):
  layer=int(g['site'][-1])+1;mlp=model.blocks[layer].mlp;assert not mlp.config.gated;L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']];b=d['captures'][gi,'native'][layer,'mlp_raw'].double().cuda();rad=(b.square().mean(-1,keepdim=True)+eps).sqrt();u=b/rad;rmsreplay.append(float((u-d['captures'][gi,'native'][layer,'mlp_normalized'].double().cuda()).norm()/u.norm()))
  def normdiff(base,v):
   rr=(base.square().mean(-1,keepdim=True)+eps).sqrt();return v/rr-base*(base*v).mean(-1,keepdim=True)/rr.pow(3)
  def bilin(uu,q):return F.linear(F.linear(uu,L)*F.linear(q,R)+F.linear(q,L)*F.linear(uu,R),D)
  def module(x):
   uu=x/(x.square().mean(-1,keepdim=True)+eps).sqrt();return F.linear(F.linear(uu,L)*F.linear(uu,R),D)
  for mode in ['block','random']:
   v=d['captures'][gi,mode][layer,'mlp_raw'].double().cuda();grid=torch.stack([bilin(u[k].expand_as(v),normdiff(b[k].expand_as(v),v)) for k in range(4)]);own=torch.stack([grid[k,k] for k in range(4)]);ref=d['captures'][gi,mode][layer,'mlp'].double().cuda();err=(own-ref).norm(dim=1)/ref.norm(dim=1).clamp_min(1e-30);replays.extend(err.cpu().tolist());h=1e-4;finite=(module(b+h*v)-module(b-h*v))/(2*h);fd.extend(((finite-own).norm(dim=1)/own.norm(dim=1).clamp_min(1e-30)).cpu().tolist());captures[gi,mode]=grid.cpu()
   arms={'own':[cos(own[a],own[c]) for a,c in itertools.combinations(range(4),2)],'common_baseline':[cos(grid[k,a],grid[k,c]) for k in range(4) for a,c in itertools.combinations(range(4),2)],'common_input':[cos(grid[a,k],grid[c,k]) for k in range(4) for a,c in itertools.combinations(range(4),2)]}
   # For each fixed raw input, hold bilinear state u fixed or normalizer derivative fixed, averaging all references.
   normonly=[];bilinonly=[]
   for k in range(4):
    q=normdiff(b,v[k].expand_as(b))
    for reference in range(4):
     no=bilin(u[reference].expand_as(u),q);bo=bilin(u,q[reference].expand_as(q));normonly.extend(cos(no[a],no[c]) for a,c in itertools.combinations(range(4),2));bilinonly.extend(cos(bo[a],bo[c]) for a,c in itertools.combinations(range(4),2))
   arms['normalizer_state_only']=normonly;arms['bilinear_state_only']=bilinonly
   for arm,cc in arms.items():records.append({'group':gi,'site':g['site'],'block':g['block'],'mode':mode,'arm':arm,'cosines':cc,'mean_cosine':float(np.mean(cc)),'median_cosine':float(np.median(cc))})
   captures[gi,mode,'validation']={'own_relative_errors':err.cpu().tolist(),'own_cosines':[cos(a,c) for a,c in zip(own,ref)]}
  print('DONE',gi,flush=True)
 summary=[]
 for site in ['residual0','residual1']:
  for mode in ['block','random']:
   summary.append({'site':site,'mode':mode,**{arm:float(np.mean([r['mean_cosine'] for r in records if r['site']==site and r['mode']==mode and r['arm']==arm])) for arm in arms}})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_normalized_replay_max_relative':max(rmsreplay),'own_derivative_replay_median_relative':float(np.median(replays)),'own_derivative_replay_max_relative':max(replays),'finite_difference_max_relative':max(fd),'whole_model_forwards':0,'test_evaluations':0,'finite':all(np.isfinite(r['mean_cosine']) for r in records)};print(audits,flush=True);assert audits['hashes_unchanged'] and audits['finite'] and max(rmsreplay)<1e-5 and max(fd)<1e-6 and max(replays)<.05;torch.save({'responses':captures,'groups':d['groups']},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':main()
