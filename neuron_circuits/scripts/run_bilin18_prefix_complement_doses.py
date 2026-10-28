"""H617 frozen subspace/complement factorial intervention across doses."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_complement_doses'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time()
 paths={'source':ART/'bilin18_prefix_frozen_transfer_captures.pt','manifest':ART/'bilin18_prefix_frozen_transfer_manifest.json','basis':ART/'bilin18_prefix_joint_conditional_locked_bases.pt','runner':Path(__file__)}
 hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};d=torch.load(paths['source'],weights_only=False);rows=json.loads(paths['manifest'].read_text());ex={r['id']:r for r in rows};B=torch.load(paths['basis'],weights_only=False)['bases']['joint'][:64];doses=[.25,.5,1.,2.]
 cfg=dict(experiment='H617',hashes=hashes,doses=doses,design='Frozen H615 joint64 and H616 full contextual interactions. Subtract in-span P, complement Q=I-P, or full I at four doses on combined recipients; native baseline. No fit/selection. Compare downstream vector/logit nonadditivity, actual full logits versus sum of separate logit shifts, native-reference KL and CE.',scope='Same copying panel; no task necessity or additive causal-share claim. KL itself is nonlinear; logit additivity is evaluated independently.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 att=model.blocks[5].attn;vec={};ortho=[];summation=[];jobs=defaultdict(list)
 for k,x in d['components'].items():
  p=(x@B.T)@B;q=x-p;vec[k]={'full':x,'span':p,'complement':q};ortho.append(float(abs((p.double()*q.double()).sum())/(p.norm()*q.norm())));summation.append(float((p+q-x).norm()/x.norm()));jobs[len(ex[k]['ids'])].append(k)
 def forward(kk,mode=None,dose=0):
  cap={};hs=[]
  def target(m,args):cap['target']=args[0][:,-1].cpu()
  hs.append(model.blocks[11].mlp.register_forward_pre_hook(target))
  if mode:
   delta=torch.stack([vec[k][mode]*dose for k in kk])
   def edit(m,args,out):y,bus=out;y=y.clone();y[:,4:]-=delta.to(y);return y,bus
   hs.append(att.register_forward_hook(edit))
  try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in kk],device='cuda'))
  finally:
   for h in hs:h.remove()
  return z.cpu(),cap['target']
 native={};replay=[]
 for kk in jobs.values():
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];z,t=forward(bb)
   for i,k in enumerate(bb):native[k]=(z[i],t[i]);replay.append(float((z[i]-d['native'][k]).abs().max()))
 result=dict(config=cfg,interventions=[],factorial=[],summary={},factorial_summary={},status='running')
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 def kl(a,b):a=a.double().log_softmax(0);b=b.double().log_softmax(0);return float((a.exp()*(a-b)).sum())
 for dose in doses:
  outs={}
  for mode in ['full','span','complement']:
   outs[mode]={}
   for kk in jobs.values():
    for st in range(0,len(kk),16):
     bb=kk[st:st+16];z,t=forward(bb,mode,dose)
     for i,k in enumerate(bb):
      zz=z[i];tt=t[i];nz,nt=native[k];e=ex[k];ans=e['groups'][e['variant']][0];outs[mode][k]=(zz,tt);fz,ft=outs['full'][k]
      result['interventions'].append(dict(id=k,wording=e['wording'],mode=mode,dose=dose,native_kl=kl(nz,zz),full_reference_kl=kl(fz,zz),omission_kl=kl(fz,nz),target_shift_norm=float((tt-nt).norm()),target_error_to_full=float((tt-ft).norm()/(ft-nt).norm()),target_ce_delta=float(nz.double().log_softmax(0)[ans]-zz.double().log_softmax(0)[ans]),full_vocab_correct=measure(zz,e['groups'])['top1'][e['variant']]))
   save();print('COMPLETE',dose,mode,flush=True)
  for k,(nz,nt) in native.items():
   fz,ft=outs['full'][k];pz,pt=outs['span'][k];qz,qt=outs['complement'][k];pred=pz+qz-nz;dt=ft-pt-qt+nt;dz=(fz-pred).double();base=(fz-nz).double();prob=nz.double().softmax(0)
   def weighted_norm(x):return ((x-(prob*x).sum()).square()*prob).sum().sqrt()
   result['factorial'].append(dict(id=k,wording=ex[k]['wording'],dose=dose,target_cross_relative=float(dt.norm()/(ft-nt).norm()),target_cross_norm=float(dt.norm()),logit_cross_relative=float(weighted_norm(dz)/weighted_norm(base)),logit_cross_weighted_norm=float(weighted_norm(dz)),full_logit_shift_weighted_norm=float(weighted_norm(base)),additive_prediction_kl=kl(fz,pred),full_native_kl=kl(nz,fz),target_component_cosine=float(torch.nn.functional.cosine_similarity(pt-nt,qt-nt,dim=0))))
  save()
 for w in range(8):
  for dose in doses:
   cc=[r for r in result['interventions'] if r['wording']==w and r['dose']==dose];den=np.mean([r['native_kl'] for r in cc if r['mode']=='full'])
   for mode in ['full','span','complement']:
    vv=[r for r in cc if r['mode']==mode];s={n:float(np.mean([v[n] for v in vv])) for n in ['native_kl','full_reference_kl','target_shift_norm','target_error_to_full','target_ce_delta','full_vocab_correct']};s['native_kl_relative_to_full']=s['native_kl']/den;s['full_reference_kl_relative_to_omission']=s['full_reference_kl']/float(np.mean([v['omission_kl'] for v in vv]));result['summary'][f'{w}_{dose}_{mode}']=s
   vv=[r for r in result['factorial'] if r['wording']==w and r['dose']==dose];result['factorial_summary'][f'{w}_{dose}']={n:float(np.mean([v[n] for v in vv])) for n in ['target_cross_relative','target_cross_norm','logit_cross_relative','logit_cross_weighted_norm','full_logit_shift_weighted_norm','additive_prediction_kl','full_native_kl','target_component_cosine']}
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_replay_max_logit_error=max(replay),projection_complement_cosine_max=max(ortho),component_sum_relative_max=max(summation),native_prompts=len(native),interventions=len(result['interventions']),factorial_cases=len(result['factorial']),finite=all(np.isfinite(v) for r in result['factorial'] for n,v in r.items() if isinstance(v,float)))
 result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
