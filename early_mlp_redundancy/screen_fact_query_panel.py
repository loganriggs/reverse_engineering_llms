"""M106 native competence and cached early states on the frozen factorial panel."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import load_model,trace


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    pp=V/'fact_query_panel_protocol.json';p=json.loads(pp.read_text());assert p['source_sha256']==sha(V.parent/'build_fact_query_panel.py');xp=V/'fact_query_panel_execution.json';x=json.loads(xp.read_text());assert x['source_sha256']==sha(__file__) and x['protocol_sha256']==sha(pp)
    model,receipt=load_model();ref=json.loads((V/'native_input_reference.json').read_text());assert receipt.weights_sha256==ref['checkpoint_sha256'];rows=[];states={};replays=[];seen=set();calls=0
    for item in p['rows']:
        if not item['valid']:continue
        ids=torch.tensor([item['token_ids']],device='cuda');out,cap=trace(model,ids);calls+=1;z=out[0,-1];answers=item['answer_token_ids'];scores=z.double()[answers];correct=item['correct_index'];wrong=[i for i in range(4) if i!=correct];key=(item['domain'],item['format'])
        if key not in seen:
            heads=[];handle=model.lm_head.register_forward_hook(lambda module,inputs,output:heads.append(output.detach()))
            try:model(ids,ids)
            finally:handle.remove()
            direct=30*torch.tanh(heads[0]/30);error=float((direct-out).abs().max());assert error<1e-4;calls+=1;seen.add(key);replays.append(dict(id=item['id'],max_absolute_error=error))
        rows.append(dict(id=item['id'],domain=item['domain'],format=item['format'],case=item['case'],a=item['a'],b=item['b'],query=item['query'],order=item['order'],correct_index=correct,predicted_index=int(scores.argmax()),four_way_correct=int(scores.argmax())==correct,exact_vocab_top1=int(z.argmax())==answers[correct],top_token=int(z.argmax()),margin_over_best_wrong=float(scores[correct]-scores[wrong].max()),correct_answer_probability=float(z.double().softmax(-1)[answers[correct]])))
        states[item['id']]=dict(logits=z.cpu(),ids=ids.cpu(),layers={layer:dict(input=cap[layer]['normalized_input'].cpu(),write=model.transformer.h[layer].mlp.Down(cap[layer]['products']).cpu()) for layer in [0,1,2]})
        if len(rows)%32==0:print('SCREENED',len(rows),flush=True)
    cells=[];summaries=[];gate=p['competence_gate']
    for domain in p['specs']:
        for fmt in ['completion','question_answer']:
            subset=[r for r in rows if r['domain']==domain and r['format']==fmt];local=[]
            for query in [0,1]:
                for order in [0,1]:
                    rr=[r for r in subset if r['query']==query and r['order']==order];n=len(rr);acc=sum(r['four_way_correct'] for r in rr)/n if n else 0.;top=sum(r['exact_vocab_top1'] for r in rr)/n if n else 0.;passed=n>=gate['min_valid_examples_per_cell'] and acc>=gate['min_four_way_accuracy'] and top>=gate['min_exact_vocab_top1'];cell=dict(domain=domain,format=fmt,query=query,order=order,n=n,four_way_accuracy=acc,exact_vocab_top1=top,passed=passed);cells.append(cell);local.append(cell)
            summaries.append(dict(domain=domain,format=fmt,n=len(subset),four_way_accuracy=float(np.mean([r['four_way_correct'] for r in subset])),exact_vocab_top1=float(np.mean([r['exact_vocab_top1'] for r in subset])),passed=all(c['passed'] for c in local)))
    sp=A/'emlp_m106_fact_query_states.pt';assert not sp.exists();torch.save(states,sp);out=dict(status='completed',source_sha256=sha(__file__),execution_sha256=sha(xp),protocol_sha256=sha(pp),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,cells=cells,summaries=summaries,native_replays=replays,states_path=str(sp),states_sha256=sha(sp),scope='Native competence only, all192frozen prompts, no featurizer or nativecomponent intervention. Fourway relative-choice andcanonical vocabulary-top1 reported separately. All4query/ordercells must pass; no failedcohort replacement. Earlylayer states cached for prospectivecausal study.')
    rp=V/'fact_query_panel_screen.json';assert not rp.exists();rp.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
