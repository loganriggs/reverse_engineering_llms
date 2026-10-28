"""H619 norm-matched direction audit on identical free-continuation recipients."""
import json,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
@torch.inference_mode()
def main():
 torch.set_num_threads(6);tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);paths={n:ART/f'{stem}_captures.pt' for n,stem in [('new','bilin18_prefix_response_mode_control'),('old','bilin18_prefix_diverse_transfer')]};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};d=torch.load(paths['new'],weights_only=False);old=torch.load(paths['old'],weights_only=False);B=torch.load(ART/'bilin18_prefix_joint_conditional_locked_bases.pt',weights_only=False)['bases']['joint'][:64];rows=json.loads((ART/'bilin18_prefix_response_mode_control_manifest.json').read_text());oldrows={r['id']:r for r in json.loads((ART/'bilin18_prefix_diverse_transfer_manifest.json').read_text())};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 out=[];geometry=[];generic=[tok.encode(t,add_special_tokens=False)[0] for t in [' the',' a',' not']];mapping={0:0,2:1,4:2,6:3,8:6,10:7}
 for e in rows:
  if e['format'] or not (e['prefix'] and e['lower']):continue
  k=e['id'];ok=str(mapping[e['wording']])+'-'+k.split('-',1)[1];assert e['ids']==oldrows[ok]['ids'];I=d['components'][k];O=old['components'][ok];scale=O.norm()/I.norm();N=I*scale;geometry.append(dict(id=k,scale=float(scale),cosine=float(torch.nn.functional.cosine_similarity(I.flatten(),O.flatten(),dim=0)),norm_relative_error=float(abs(N.norm()-O.norm())/O.norm())))
  for direction,X in [('new_matched',N),('old',O)]:
   P=(X@B.T)@B
   for mode,base in [('full',X),('span',P),('complement',X-P)]:
    delta=base
    def hook(m,args,out):y,b=out;y=y.clone();y[:,4:]-=delta.to(y);return y,b
    h=model.blocks[5].attn.register_forward_hook(hook)
    try:z,_=run(model,torch.tensor([e['ids']],device='cuda'))
    finally:h.remove()
    z=z[0].cpu();mm=measure(z,e['groups']);lp=z.double().log_softmax(0);np=d['native'][k].double().log_softmax(0);g=e['groups'][e['variant']];out.append(dict(id=k,wording=e['wording'],direction=direction,mode=mode,kl=float((np.exp()*(np-lp)).sum()),candidate_correct=bool(mm['margin']>0 if e['variant']==0 else mm['margin']<0),top1_correct=mm['top1'][e['variant']],answer_mass=float(lp[g].exp().sum()),generic_margin=float(torch.logsumexp(z[g].double(),0)-torch.logsumexp(z[generic].double(),0)),top_token=tok.decode([int(z.argmax())])))
 result=dict(records=out,geometry=geometry,hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(ART/'bilin18_prefix_response_mode_control_norm_audit.json').write_text(json.dumps(result,indent=2));print('complete',len(out))
if __name__=='__main__':main()
