"""M108 native factorial responses and early-layer states; no correct mixed labels."""
import json
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'repetition_cube_protocol.json';p=json.loads(pp.read_text());xp=V/'repetition_cube_execution.json';x=json.loads(xp.read_text());assert x['source_sha256']==sha(__file__) and x['protocol_sha256']==sha(pp);assert json.loads((V/'repetition_cube_algebra_toys.json').read_text())['status']=='passed'
    nr=json.loads((V/'whole_task_native.json').read_text());assert sha(nr['samples_path'])==nr['samples_sha256'];old=torch.load(nr['samples_path'],weights_only=True);model,receipt=load_model();assert receipt.weights_sha256==nr['checkpoint_sha256'];rows=[];saved={};replays=0
    for item in p['rows']:
        ids=torch.tensor([item['token_ids']],device='cuda');logits,cap=trace(model,ids);z=logits[0,-1];groups=item['answer_groups'];scores=torch.stack([torch.logsumexp(z.double()[g],0) for g in groups]);row=dict(id=item['id'],cohort=item['cohort'],variant_group_margin=float(scores[1]-scores[0]),group_choice=int(scores.argmax()),top_token=int(z.argmax()),top1_value_group=next((i for i,g in enumerate(groups) if int(z.argmax()) in g),None))
        if item['cohort']=='historical_replay':assert torch.equal(z.cpu(),old['baselines'][item['id']]['logits']);replays+=1
        else:row.update(case=item['case'],format=item['format'],vertex=item['vertex'],bits=item['bits'],canonical_margin=float(z.double()[item['canonical_answer_tokens'][1]]-z.double()[item['canonical_answer_tokens'][0]]))
        rows.append(row);saved[item['id']]=dict(logits=z.cpu(),ids=ids.cpu(),layers={layer:dict(input=cap[layer]['normalized_input'].cpu(),write=model.transformer.h[layer].mlp.Down(cap[layer]['products']).cpu()) for layer in [0,1,2]})
    path=A/'emlp_m108_repetition_cube_states.pt';assert not path.exists();torch.save(saved,path);result=dict(status='completed',source_sha256=sha(__file__),protocol_sha256=sha(pp),execution_sha256=sha(xp),checkpoint_sha256=receipt.weights_sha256,forward_count=len(rows),historical_replays=replays,rows=rows,states_path=str(path),states_sha256=sha(path));dest=V/'repetition_cube_native.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print('COMPLETED',len(rows),'REPLAYS',replays)
if __name__=='__main__':main()
