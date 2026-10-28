"""H673: matched carried-value maps across heads, with scrambled head4 controls."""
import json,hashlib,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_carried_head_geometry'
@torch.no_grad()
def main():
 torch.set_num_threads(6);t=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','native':ART/'bilin18_operation_paraphrase_panel_captures.pt','factors':ART/'bilin18_head4_value_conditioned_captures.pt','reference':ART/'bilin18_value_source_split_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['native'],weights_only=False);fac=torch.load(paths['factors'],weights_only=False);ref=torch.load(paths['reference'],weights_only=False);held=ref['heldout_indices'];model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 cfg=dict(experiment='H673',hashes=hashes,design='All9 matched layer14 head maps of the native carried input value,128 existing contexts.32 fixed seed673 signed coordinate permutations of head4 bus slice, shared across contexts, as alignment controls. Direct readout slopes in native units and per unit projected residual norm. No gate coefficients, no fitting.',limitations='Direct path only; no head necessity or full downstream comparison. Native head comparison changes both bus slice and projection. Scrambled mappings are off-manifold controls. Contexts repeat16 unique inputs, not128 independent identities.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));att=model.blocks[14].attn;gain=float(torch.stack([model.blocks[l].lambdas[0] for l in range(15,18)]).prod());bus=torch.stack([cap['bus'][i].view(-1,9,128)[fac['token_metadata'][i]['labels'].index(2)] for i in held]).double().cuda();W=att.c_proj.weight.double();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 def token(v):
  ids=tok.encode(' '+v,add_special_tokens=False);assert len(ids)==1;return ids[0]
 targets=[]
 for i in held:
  r=rows[i];other=sorted({token(q['input']) for q in rows if q['family']==r['family'] and q['input']!=r['input']});assert len(other)==7;targets.append([token(r['input']),token(r['answer'])]+other)
 ix=torch.tensor(targets,device='cuda');X=cap['native_captures'][17][held].double().cuda();U=model.lm_head.weight.double()[ix];radius=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();pre=torch.einsum('btd,bd->bt',U,X/radius);sat=1-torch.tanh(pre/30).square();grad=sat[:,:,None]*(U/radius[:,:,None]-X[:,None,:]*torch.einsum('btd,bd->bt',U,X)[:,:,None]/(1152*radius[:,:,None]**3));rng=torch.Generator().manual_seed(673);specs=[('native',h,None,None) for h in range(9)]+[('scrambled',4,torch.randperm(128,generator=rng).cuda(),(2*torch.randint(2,(128,),generator=rng)-1).double().cuda()) for _ in range(32)];records=[];directions=[];replay=0.;value_replay=0.
 for k,(kind,h,perm,sign) in enumerate(specs):
  v=bus[:,h] if perm is None else bus[:,h,perm]*sign;d=float(att.lamb)*v@W[:,h*128:(h+1)*128].T;n=d.norm(dim=-1);sl=gain*torch.einsum('btd,bd->bt',grad,d);directions.append(d.cpu())
  if kind=='native' and h==4:
   value_replay=float(((d.cpu()-ref['parts']['bus']).norm(dim=-1)/ref['parts']['bus'].norm(dim=-1)).max());truth=ref['direct']['bus'].gather(1,ix.cpu());replay=float((sl.cpu()-truth).abs().max())
  for b,i in enumerate(held):
   r=rows[i];records.append(dict(id=r['id'],family=r['family'],control=r['control'],item=r['item'],form=r['form'],kind=kind,head=h,control_index=k-9 if kind=='scrambled' else None,projected_norm=float(n[b]),current=float(sl[b,0]),answer=float(sl[b,1]),other_minus_current=float(sl[b,2:].mean()-sl[b,0]),unit_current=float(sl[b,0]/n[b]),unit_other_minus_current=float((sl[b,2:].mean()-sl[b,0])/n[b])))
 assert value_replay<1e-5 and replay<1e-5
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},head4_value_replay_max_relative=value_replay,head4_readout_replay_max_abs=replay,whole_model_forwards=0,finite=all(torch.isfinite(torch.tensor([r['current'],r['unit_current']])).all().item() for r in records),contexts=len(held),unique_input_strings=len({rows[i]['input'] for i in held}),scramble_seed=673)
 torch.save(dict(directions=torch.stack(directions),heldout_indices=held),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-t),indent=2));print(audits)
if __name__=='__main__':main()
