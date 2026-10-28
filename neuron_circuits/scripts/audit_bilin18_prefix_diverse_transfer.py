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
 torch.set_num_threads(6);tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);d=torch.load(ART/'bilin18_prefix_diverse_transfer_captures.pt',weights_only=False);B=torch.load(ART/'bilin18_prefix_joint_conditional_locked_bases.pt',weights_only=False)['bases']['joint'][:64];rows=json.loads((ART/'bilin18_prefix_diverse_transfer_manifest.json').read_text());model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 out=[]
 for e in rows:
  if e['family']!='capital_recall' or not (e['prefix'] and e['lower']):continue
  k=e['id'];I=d['components'][k];P=(I@B.T)@B
  for mode,delta in [('native',I*0),('full',I),('span',P),('complement',I-P)]:
   def hook(m,args,out):y,b=out;y=y.clone();y[:,4:]-=delta.to(y);return y,b
   h=model.blocks[5].attn.register_forward_hook(hook)
   try:z,_=run(model,torch.tensor([e['ids']],device='cuda'))
   finally:h.remove()
   z=z[0].cpu();mm=measure(z,e['groups']);idx=z.topk(5).indices.tolist();out.append(dict(id=k,wording=e['wording'],mode=mode,top5=[tok.decode([t]) for t in idx],margin=mm['margin'],candidate_correct=(mm['margin']>0 if e['variant']==0 else mm['margin']<0),top1_correct=mm['top1'][e['variant']],intended_token_ce=float(-z.double().log_softmax(0)[e['groups'][e['variant']][0]])))
 (ART/'bilin18_prefix_diverse_transfer_capital_audit.json').write_text(json.dumps(dict(runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=out),indent=2))
 print('complete',len(out))
if __name__=='__main__':main()
