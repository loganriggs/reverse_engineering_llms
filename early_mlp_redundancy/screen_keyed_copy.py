"""M107 frozen native screen with exact historical-copy replay."""
import json
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'keyed_copy_protocol.json';p=json.loads(pp.read_text());xp=V/'keyed_copy_execution.json';x=json.loads(xp.read_text());assert x['source_sha256']==sha(__file__) and x['protocol_sha256']==sha(pp);assert p['source_sha256']==sha(V.parent/'build_keyed_copy_challenge.py')
    nr=json.loads((V/'whole_task_native.json').read_text());assert sha(nr['samples_path'])==nr['samples_sha256'];native=torch.load(nr['samples_path'],weights_only=True);model,receipt=load_model();assert receipt.weights_sha256==nr['checkpoint_sha256'];rows=[];samples={};replays=0
    for item in p['rows']:
        ids=torch.tensor([item['token_ids']],device='cuda');out,cap=trace(model,ids);z=out[0,-1];scores=torch.stack([torch.logsumexp(z.double()[g],0) for g in item['answer_groups']]);pred=int(scores.argmax());correct=item['correct_index'];wrong=[j for j in range(len(scores)) if j!=correct];row={k:v for k,v in item.items() if k not in ['token_ids','prompt','answer_groups']};row.update(predicted_index=pred,choice_correct=pred==correct,vocab_top1=int(z.argmax()) in item['answer_groups'][correct],top_token=int(z.argmax()),margin_over_best_wrong=float(scores[correct]-scores[wrong].max()));rows.append(row)
        if item['cohort']=='historical_replay':assert torch.equal(z.cpu(),native['baselines'][item['id']]['logits']);replays+=1
        samples[item['id']]=dict(logits=z.cpu(),ids=ids.cpu(),layers={layer:dict(input=cap[layer]['normalized_input'].cpu(),write=model.transformer.h[layer].mlp.Down(cap[layer]['products']).cpu()) for layer in [0,1,2]})
    cells=[];summaries=[];gate=p['gate']
    for fmt in ['semicolon','comma']:
        rr=[x for x in rows if x.get('format')==fmt];local=[]
        for query in [0,1]:
            for order in [0,1]:
                subset=[x for x in rr if x['query']==query and x['order']==order];n=len(subset);acc=sum(x['choice_correct'] for x in subset)/n;top=sum(x['vocab_top1'] for x in subset)/n;cell=dict(format=fmt,query=query,order=order,n=n,choice_accuracy=acc,vocab_top1=top,passed=n>=gate['min_n_per_query_order_cell'] and acc>=gate['min_four_way_accuracy'] and top>=gate['min_vocab_top1']);cells.append(cell);local.append(cell)
        summaries.append(dict(format=fmt,n=len(rr),choice_accuracy=sum(x['choice_correct'] for x in rr)/len(rr),vocab_top1=sum(x['vocab_top1'] for x in rr)/len(rr),passed=all(x['passed'] for x in local)))
    path=A/'emlp_m107_keyed_copy_states.pt';assert not path.exists();torch.save(samples,path);out=dict(status='completed',source_sha256=sha(__file__),protocol_sha256=sha(pp),execution_sha256=sha(xp),checkpoint_sha256=receipt.weights_sha256,forward_count=len(rows),historical_replays=replays,rows=rows,cells=cells,summaries=summaries,states_path=str(path),states_sha256=sha(path));f=V/'keyed_copy_screen.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2),'REPLAY',replays)
if __name__=='__main__':main()
