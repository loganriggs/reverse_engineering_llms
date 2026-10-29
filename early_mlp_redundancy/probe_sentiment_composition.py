"""M115 native crossed sentiment/negation screen with exact historical replays."""
import json
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace


def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'sentiment_composition_protocol.json';p=json.loads(pp.read_text());assert p['source_sha256']==sha(V.parent/'build_sentiment_composition.py');xp=V/'sentiment_composition_execution.json';x=json.loads(xp.read_text());assert x['source_sha256']==sha(__file__) and x['protocol_sha256']==sha(pp)
 nr=json.loads((V/'training_causal_coverage.json').read_text());assert sha(nr['samples_path'])==nr['samples_sha256'];old=torch.load(nr['samples_path'],weights_only=True);model,mr=load_model();assert mr.weights_sha256==nr['checkpoint_sha256'];rows=[];saved={};replays=0
 for item in p['rows']:
  ids=torch.tensor([item['token_ids']],device='cuda');out,cap=trace(model,ids);z=out[0,-1];scores=torch.stack([torch.logsumexp(z.double()[g],0) for g in item['answer_groups']]);choice=int(scores.argmax());top=int(z.argmax());row=dict(id=item['id'],cohort=item['cohort'],expected_group=item['expected_group'],chosen_group=choice,group_correct=choice==item['expected_group'],top1_correct=top in item['answer_groups'][item['expected_group']],top1_token=top,positive_minus_negative_margin=float(scores[0]-scores[1]))
  if item['cohort']=='historical_replay':assert torch.equal(z.cpu(),old['baselines'][item['id']]['logits']);replays+=1
  else:row.update({k:item[k] for k in ['format','lexicon','subject','valence','negated']})
  rows.append(row);saved[item['id']]=dict(logits=z.cpu(),ids=ids.cpu(),layers={layer:dict(input=cap[layer]['normalized_input'].cpu(),write=model.transformer.h[layer].mlp.Down(cap[layer]['products']).cpu()) for layer in [0,1,2]})
 sp=A/'emlp_m115_sentiment_composition.pt';assert not sp.exists();torch.save(saved,sp);dest=V/'sentiment_composition_native.json';assert not dest.exists();result=dict(status='completed_pending_audit',source_sha256=sha(__file__),protocol_sha256=sha(pp),execution_sha256=sha(xp),checkpoint_sha256=mr.weights_sha256,forwards=len(rows),exact_historical_replays=replays,rows=rows,path=str(sp),sha256=sha(sp));dest.write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',len(rows),'REPLAYS',replays)
if __name__=='__main__':main()
