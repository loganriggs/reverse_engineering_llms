"""H673: matched carried-value maps across heads, with scrambled head4 controls."""
import json,hashlib,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_native_input_head_balance'
@torch.no_grad()
def main():
 torch.set_num_threads(6);t=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','native':ART/'bilin18_operation_paraphrase_panel_captures.pt','factors':ART/'bilin18_head4_value_conditioned_captures.pt','reference':ART/'bilin18_value_source_split_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['native'],weights_only=False);fac=torch.load(paths['factors'],weights_only=False);ref=torch.load(paths['reference'],weights_only=False);held=ref['heldout_indices'];model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 cfg=dict(experiment='H674',hashes=hashes,design='Native signed input-token coefficients times separate weighted local and carried values, all9 heads layer14,128 existing contexts. Direct residual/readout derivatives. Reconstruct full attention output over all positions and compare saved head4 coefficients.',limitations='Local direct-path accounting, not a finite removal or causal-share analysis. Input position only in reported head contributions. No task correctness inference; existing16 unique input strings.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));att=model.blocks[14].attn;gain=float(torch.stack([model.blocks[l].lambdas[0] for l in range(15,18)]).prod());bus=torch.stack([cap['bus'][i].view(-1,9,128)[fac['token_metadata'][i]['labels'].index(2)] for i in held]).double().cuda();W=att.c_proj.weight.double();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 def token(v):
  ids=tok.encode(' '+v,add_special_tokens=False);assert len(ids)==1;return ids[0]
 targets=[]
 for i in held:
  r=rows[i];other=sorted({token(q['input']) for q in rows if q['family']==r['family'] and q['input']!=r['input']});assert len(other)==7;targets.append([token(r['input']),token(r['answer'])]+other)
 ix=torch.tensor(targets,device='cuda');X=cap['native_captures'][17][held].double().cuda();U=model.lm_head.weight.double()[ix];radius=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();pre=torch.einsum('btd,bd->bt',U,X/radius);sat=1-torch.tanh(pre/30).square();grad=sat[:,:,None]*(U/radius[:,:,None]-X[:,None,:]*torch.einsum('btd,bd->bt',U,X)[:,:,None]/(1152*radius[:,:,None]**3))
 import torch.nn.functional as F
 block=model.blocks[14];records=[];vectors=[];coeffs=[];replays=[];coefreplay=[]
 for b,i in enumerate(held):
  x=cap['caches'][14][i][None].cuda();x0=cap['x0'][i][None].cuda();vb=cap['bus'][i][None].cuda();y=F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,));B,T,_=y.shape;p=fac['token_metadata'][i]['labels'].index(2)
  q,k,q2,k2=[m(y).view(B,T,9,128) for m in [att.c_q,att.c_k,att.c_q2,att.c_k2]];cos,sin=att.rotary(q);q,k,q2,k2=[model.tt.apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin) for v in [q,k,q2,k2]];coef=((q[:,-1,None]*k).sum(-1)/128)*((q2[:,-1,None]*k2).sum(-1)/128);coeffs.append(coef[0,p].cpu());coefreplay.append(abs(float(coef[0,p,4])-float(fac['coefficients'][i][p])))
  vals={'local':(1-att.lamb)*att.c_v(y).view(B,T,9,128),'bus':att.lamb*vb.view(B,T,9,128)};weighted={name:coef[...,None].double()*v.double() for name,v in vals.items()};effective=vals['local']+vals['bus'];preproj=(coef[...,None]*effective).sum(1).reshape(B,1152);truth,_=att(y,vb);estimated=F.linear(preproj,att.c_proj.weight);replays.append(float((estimated[0]-truth[0,-1]).norm()/truth[0,-1].norm()))
  vv=[]
  for name,values in weighted.items():
   hv=torch.stack([values[0,p,h]@W[:,h*128:(h+1)*128].T for h in range(9)]);vv.append(hv.cpu());sl=gain*torch.einsum('td,hd->ht',grad[b],hv)
   for h in range(9):
    row=rows[i];records.append(dict(id=row['id'],family=row['family'],control=row['control'],item=row['item'],form=row['form'],head=h,component=name,coefficient=float(coef[0,p,h]),current=float(sl[h,0]),answer=float(sl[h,1]),other_minus_current=float(sl[h,2:].mean()-sl[h,0])))
  vectors.append(torch.stack(vv))
 assert max(replays)<1e-4 and max(coefreplay)<1e-4
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},attention_output_replay_max_relative=max(replays),head4_coefficient_replay_max_abs=max(coefreplay),whole_model_forwards=0,cached_attention_evaluations=len(held),finite=all(torch.isfinite(torch.tensor([r['current'],r['answer']])).all().item() for r in records))
 torch.save(dict(vectors=torch.stack(vectors),component_order=['local','bus'],coefficients=torch.stack(coeffs),heldout_indices=held),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-t),indent=2));print(audits)
if __name__=='__main__':main()
