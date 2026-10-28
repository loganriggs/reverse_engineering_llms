import json,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
from audit_bilin18_mlp_discriminators import TOKENIZER
@torch.inference_mode()
def main():
 torch.set_num_threads(6);tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);rows=[e for e in json.loads((ART/'bilin18_emission_surface_transfer_manifest.json').read_text()) if e['family']=='plural'];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 groups=defaultdict(list)
 for e in rows:groups[len(e['ids'])].append(e)
 out=[]
 for rr in groups.values():
  seq=torch.tensor([e['ids'] for e in rr],device='cuda');base=seq.shape[1]
  for step in range(8):z,_=run(model,seq);seq=torch.cat([seq,z.argmax(-1,keepdim=True)],dim=1)
  for e,x in zip(rr,seq[:,base:].cpu().tolist()):out.append(dict(id=e['id'],prompt=e['prompt'],expected=e['answers'][e['item']],generated_ids=x,generated_text=tok.decode(x)))
 (ART/'bilin18_emission_surface_transfer_native_generations.json').write_text(json.dumps(dict(records=out,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
