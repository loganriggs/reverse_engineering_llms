"""H668 diagnostic: direction-specific coefficients and finite coefficient swaps."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_direction_gate'
@torch.no_grad()
def main():
 torch.set_num_threads(6);paths={'manifest':ART/'bilin18_direction_matched_panel_manifest.json','captures':ART/'bilin18_direction_matched_panel_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['captures'],weights_only=False);lookup={(r['family'],r['item'],r['control'],r['form']):i for i,r in enumerate(rows)};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);pos={};buckets=defaultdict(list)
 for i,r in enumerate(rows):
  enc=tok(r['prompt'],add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==r['ids'];lo=r['prompt'].index('\nInput: ')+len('\nInput: ');hi=lo+len(r['input']);pp=[k for k,(a,b) in enumerate(enc['offset_mapping']) if any(lo<=p<hi and not r['prompt'][p].isspace() for p in range(a,b))];assert len(pp)==1;pos[i]=pp[0];buckets[len(r['ids'])].append(i)
 coeff=torch.empty(len(rows),dtype=torch.float64);value=torch.empty(len(rows),1152,dtype=torch.float64);block=model.blocks[14];att=block.attn;W=att.c_proj.weight[:,512:640].double();reconstruction=[]
 for ii in buckets.values():
  for st in range(0,len(ii),16):
   bb=ii[st:st+16];x=torch.stack([cap['caches'][14][i] for i in bb]).cuda();x0=torch.stack([cap['x0'][i] for i in bb]).cuda();bus=torch.stack([cap['bus'][i] for i in bb]).cuda();y=F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,));B,T,_=y.shape;q,k,q2,k2=[m(y).view(B,T,9,128) for m in [att.c_q,att.c_k,att.c_q2,att.c_k2]];cos,sin=att.rotary(q);q,k,q2,k2=[model.tt.apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin) for v in [q,k,q2,k2]];cc=((q[:,-1,4,None]*k[:,:,4]).sum(-1)/128)*((q2[:,-1,4,None]*k2[:,:,4]).sum(-1)/128);v=(1-att.lamb)*att.c_v(y).view(B,T,9,128)+att.lamb*bus.view(B,T,9,128);vv=F.linear(v[:,:,4].double(),W);saved=[]
   def capture(m,args):saved.append(args[0][:,-1,512:640].clone())
   h=att.c_proj.register_forward_pre_hook(capture)
   try:att(y,bus)
   finally:h.remove()
   truth=F.linear(saved[0].double(),W);reconstruction.append(float(((vv*cc.double().unsqueeze(-1)).sum(1)-truth).norm(dim=-1).div(truth.norm(dim=-1)).max()))
   for n,i in enumerate(bb):coeff[i]=cc[n,pos[i]].double().cpu();value[i]=vv[n,pos[i]].cpu()
 means={};summary=[]
 for f in ['month','number']:
  for form in range(4):
   for control in ['next','previous']:
    ii=[i for i,r in enumerate(rows) if r['family']==f and r['control']==control and r['form']==form and r['split']=='discovery'];delta=torch.stack([coeff[i]-coeff[lookup[f,rows[i]['item'],'copy',form]] for i in ii]);means[f,form,control]=delta.mean();summary.append(dict(family=f,form=form,control=control,discovery_delta=float(delta.mean())))
 held=[i for i,r in enumerate(rows) if r['control'] in ['next','previous'] and r['split']=='transfer'];geometry=[]
 for i in held:
  r=rows[i];f,form,c=r['family'],r['form'],r['control'];copy=lookup[f,r['item'],'copy',form];truth=coeff[i]-coeff[copy]
  for label,pred in [('own',means[f,form,c]),('opposite',means[f,form,'previous' if c=='next' else 'next'])]:geometry.append(dict(id=r['id'],family=f,control=c,mode=label,error_squared=float((pred-truth).square()),target_squared=float(truth.square())))
 # Hold out whole items; patch only head4 input coefficient while retaining receiver values.
 hb=defaultdict(list)
 for i in held:hb[len(rows[i]['ids'])].append(i)
 outputs={};records=[];forwards=0;replay=0.
 for mode in ['native','mean_change','exact_change']:
  zz={}
  for ii in hb.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];delta=[]
    for i in bb:
     r=rows[i];op='previous' if r['control']=='next' else 'next';d=lookup[r['family'],r['item'],op,r['form']];dc=coeff[d]-coeff[i] if mode=='exact_change' else means[r['family'],r['form'],op]-means[r['family'],r['form'],r['control']];delta.append(dc*value[i])
    patch=torch.stack(delta);hooks=[]
    if mode!='native':
     def edit(m,args,out):y,b=out;y=y.clone();y[:,-1]+=patch.to(y);return y,b
     hooks.append(model.blocks[14].attn.register_forward_hook(edit))
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
    for n,i in enumerate(bb):zz[i]=z[n].cpu()
  outputs[mode]=zz
  for i in held:
   r=rows[i];op='previous' if r['control']=='next' else 'next';d=lookup[r['family'],r['item'],op,r['form']];lp=zz[i].double().log_softmax(-1);tl=cap['native_logits'][d].double().log_softmax(-1);p=tl.exp();aid=rows[d]['answer_id'];cs=rows[d]['candidate_ids'];records.append(dict(id=r['id'],family=r['family'],direction=r['control']+'_to_'+op,form=r['form'],mode=mode,opposite_kl=float((p*(tl-lp)).sum()),opposite_ce=float(-lp[aid]),candidate_correct=cs[int(zz[i][cs].argmax())]==aid,top1_correct=int(zz[i].argmax())==aid))
   if mode=='native':replay=max(replay,float((zz[i]-cap['native_logits'][i]).abs().max()))
 assert replay<1e-3 and max(reconstruction)<1e-5;audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,cached_attention_evaluations=len(rows),head_reconstruction_max_relative=max(reconstruction),native_replay_max_abs=replay,heldout_rows=len(held),finite=all(np.isfinite(r['opposite_kl']) for r in records));torch.save(dict(coefficients=coeff,values=value,means=means,outputs=outputs),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',coefficient_summary=summary,geometry=geometry,records=records,audits=audits),indent=2));print('AUDITS',audits)
if __name__=='__main__':main()
