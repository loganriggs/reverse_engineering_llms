"""H671: direct residual/readout alignment versus complete finite downstream effects."""
import json,time,hashlib
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_direct_value_readout'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','native':ART/'bilin18_operation_paraphrase_panel_captures.pt','factors':ART/'bilin18_head4_value_conditioned_captures.pt','finite':ART/'bilin18_identity_suppression_domains_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());native=torch.load(paths['native'],weights_only=False);factors=torch.load(paths['factors'],weights_only=False);finite=torch.load(paths['finite'],weights_only=False);held=finite['heldout_indices'];V=torch.stack([factors['projected_values'][i][[k for k,v in enumerate(factors['token_metadata'][i]['labels']) if v==2][0]] for i in held]).cuda();model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 gain=float(torch.stack([model.blocks[l].lambdas[0] for l in range(15,18)]).prod());W=model.lm_head.weight.double();X=native['native_captures'][17][held].double().cuda();radius=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();D=gain*V;sat=1-torch.tanh((X/radius)@W.T/30).square();raw=D@W.T;direct=sat*((D/radius-X*(D*X).mean(-1,keepdim=True)/radius.pow(3))@W.T);full=((finite['logits']['input',.05].double()-finite['logits']['input',-.05].double())/.1).cuda();fullsmall=((finite['logits']['input',.025].double()-finite['logits']['input',-.025].double())/.05).cuda();P=native['native_logits'][held].double().softmax(-1).cuda()
 def read(x):return 30*torch.tanh((x/(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt())@W.T/30)
 def center(z):return z-(z*P).sum(-1,keepdim=True)
 def norm(z):return (center(z).square()*P).sum(-1).sqrt()
 directfinite=(read(X+.025*D)-read(X-.025*D))/.05;linearerr=norm(directfinite-direct)/norm(direct);fullstep=norm(full-fullsmall)/norm(fullsmall);replay=float((read(X).cpu()-native['native_logits'][held].double()).abs().max());assert replay<1e-3 and bool(torch.isfinite(direct).all());err=norm(full-direct)/norm(full);cos=(center(full)*center(direct)*P).sum(-1)/(norm(full)*norm(direct));ratio=norm(direct)/norm(full);tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);records=[]
 def token(x):
  ids=tok.encode(' '+x,add_special_tokens=False);assert len(ids)==1;return ids[0]
 for k,i in enumerate(held):
  r=rows[i];cur=token(r['input']);ans=token(r['answer']);others=sorted({token(v['input']) for v in rows if v['family']==r['family'] and v['input']!=r['input']});entry=dict(id=r['id'],family=r['family'],control=r['control'],form=r['form'],item=r['item'],direct_full_relative_error=float(err[k]),direct_full_weighted_cosine=float(cos[k]),direct_full_norm_ratio=float(ratio[k]),direct_linearization_error=float(linearerr[k]),full_step_error=float(fullstep[k]))
  for name,z in [('raw_projection',raw),('direct',direct),('full',full),('downstream_difference',full-direct)]:entry[name]=dict(current=float(z[k,cur]),answer=float(z[k,ans]),other=float(z[k,others].mean()),answer_minus_current=float(z[k,ans]-z[k,cur]),other_minus_current=float(z[k,others].mean()-z[k,cur]))
  records.append(entry)
 cfg=dict(experiment='H671',hashes=hashes,design='Same128 H670 contexts. Direct skip direction equals input-value vector times lambdas0 of blocks15-17. Analytic final RMS/tanh readout derivative at native final state versus H670 central finite full downstream slope at+/-0.05. Raw unembedding projection also reported. Weighted centered full-vocabulary metrics; analytic derivative checked against final-readout-only finite differences+/-0.025.',limitations='Direct-path diagnostic omits branches by construction, not a finite branch ablation. Full-minus-direct approximates combined downstream contribution near zero; not causal shares. Raw projection lacks final normalization/saturation and is only alignment evidence.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},new_whole_model_forwards=0,residual_skip_gain=gain,native_readout_replay_max_abs=replay,direct_linearization_max_relative=float(linearerr.max()),full_step_max_relative=float(fullstep.max()),finite=all(np.isfinite(r['direct_full_relative_error']) for r in records));torch.save(dict(direct=direct.cpu(),raw=raw.cpu(),heldout_indices=held,gain=gain),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits)
if __name__=='__main__':main()
