"""Independent factorial labels, token alignment, logits and competence gates."""
import json,itertools
import numpy as np
import torch,tiktoken
from fit_pca_encoder import V,sha


def main():
    torch.set_num_threads(1);pp=V/'fact_query_panel_protocol.json';rp=V/'fact_query_panel_screen.json';p=json.loads(pp.read_text());r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(pp) and r['source_sha256']==sha(V.parent/'screen_fact_query_panel.py');assert sha(r['states_path'])==r['states_sha256'];saved=torch.load(r['states_path'],weights_only=True);tok=tiktoken.get_encoding('gpt2');meta={x['id']:x for x in p['rows']};rows={x['id']:x for x in r['rows']};assert len(meta)==len(rows)==len(saved)==192;maximum=0.;checks=0
    def check(a,b):
        nonlocal maximum,checks
        assert np.isclose(a,b,atol=1e-10,rtol=1e-10),(a,b);maximum=max(maximum,abs(a-b));checks+=1
    grid={(x['domain'],x['format'],x['case'],x['a'],x['b'],x['query'],x['order']) for x in p['rows']};expected=set(itertools.product(p['specs'],['completion','question_answer'],[0,1],[0,1],[0,1],[0,1],[0,1]));assert grid==expected
    old=set()
    for name,h in p['prior_protocol_hashes'].items():
        assert sha(V/name)==h
        for pair in json.loads((V/name).read_text())['pairs']:
            for ids in pair['token_ids']:old.add(tuple(ids))
    for key,m in meta.items():
        assert m['valid'] and tok.encode(m['prompt'])==m['token_ids'] and saved[key]['ids'][0].tolist()==m['token_ids'];a,b=m['a'],m['b'];assignments=[m['values'][a],m['values'][2+b]];assert assignments==m['assignments'];correct=m['values'].index(assignments[m['query']]);assert correct==m['correct_index']==rows[key]['correct_index']
        spec=p['specs'][m['domain']];facts=[spec['fact'].format(name=name,value=value) for name,value in zip(m['names'],assignments)];facts=facts if m['order']==0 else facts[::-1];prefix=' '.join(facts);query_name=m['names'][m['query']]
        prompt=prefix+' '+spec['completion'].format(name=query_name) if m['format']=='completion' else prefix+'\nQuestion: '+spec['question'].format(name=query_name)+'\n'+spec['answer_prefix'];assert prompt==m['prompt']
        for value,answer in zip(m['values'],m['answer_token_ids']):assert tok.encode(prompt+' '+value)==m['token_ids']+[answer]
        assert len(set(m['answer_token_ids']))==4;assert m['exact_historical_prompt_overlap']==(tuple(m['token_ids']) in old)
        role='potential_training' if m['case']==0 and m['format']=='completion' and a==b else 'reserved_combination_or_context';assert m['data_role']==role
        z=saved[key]['logits'].double().numpy();row=rows[key];choices=z[m['answer_token_ids']];pred=int(np.argmax(choices));assert row['predicted_index']==pred and row['four_way_correct']==(pred==correct);top=int(np.argmax(z));assert row['top_token']==top and row['exact_vocab_top1']==(top==m['answer_token_ids'][correct]);wrong=[choices[j] for j in range(4) if j!=correct];check(float(choices[correct]-max(wrong)),row['margin_over_best_wrong']);weights=np.exp(z-z.max());check(float(weights[m['answer_token_ids'][correct]]/weights.sum()),row['correct_answer_probability'])
        for layer in [0,1,2]:
            item=saved[key]['layers'][layer];assert item['input'].shape==item['write'].shape==(1,len(m['token_ids']),1152) and torch.isfinite(item['input']).all() and torch.isfinite(item['write']).all()
    edgekeys=set();aligned={k:0 for k in ['a','b','query','order']}
    for e in p['edges']:
        a=meta[e['recipient_id']];b=meta[e['donor_id']];factor=e['factor'];assert a[factor]==0 and b[factor]==1
        for field in ['domain','format','case','a','b','query','order']:
            if field!=factor:assert a[field]==b[field]
        equal=len(a['token_ids'])==len(b['token_ids']);assert e['same_token_length']==equal;aligned[factor]+=equal;positions=[i for i,(x,y) in enumerate(zip(a['token_ids'],b['token_ids'])) if x!=y] if equal else None;assert positions==e['changed_positions'];assert e['answer_should_change']==(a['values'][a['correct_index']]!=b['values'][b['correct_index']]);edgekeys.add((a['id'],factor))
    assert len(edgekeys)==384 and edgekeys=={(m['id'],factor) for m in meta.values() for factor in aligned if m[factor]==0}
    gate=p['competence_gate']
    for cell in r['cells']:
        rr=[x for x in r['rows'] if all(x[k]==cell[k] for k in ['domain','format','query','order'])];assert len(rr)==cell['n']==8;acc=sum(x['four_way_correct'] for x in rr)/8;top=sum(x['exact_vocab_top1'] for x in rr)/8;check(acc,cell['four_way_accuracy']);check(top,cell['exact_vocab_top1']);assert cell['passed']==(acc>=gate['min_four_way_accuracy'] and top>=gate['min_exact_vocab_top1'])
    for summary in r['summaries']:
        rr=[x for x in r['rows'] if x['domain']==summary['domain'] and x['format']==summary['format']];cells=[x for x in r['cells'] if x['domain']==summary['domain'] and x['format']==summary['format']];assert len(rr)==summary['n']==32 and len(cells)==4;check(sum(x['four_way_correct'] for x in rr)/32,summary['four_way_accuracy']);check(sum(x['exact_vocab_top1'] for x in rr)/32,summary['exact_vocab_top1']);assert summary['passed']==all(x['passed'] for x in cells)
    assert len(r['native_replays'])==6 and r['forward_count']==198
    out=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(pp),result_sha256=sha(rp),prompts=192,edges=384,aligned_edges=aligned,metric_checks=checks,max_metric_absolute_error=maximum,passed_cohorts=sum(x['passed'] for x in r['summaries']),potential_training_prompts=sum(x['data_role']=='potential_training' for x in meta.values()),scope='All labels/factorialcells/answer tokenizations/prompt reconstruction andsplits checked; allfinal logits independently rescored. Allcompetence decisions andfactor edgepositions checked. Captured earlystates checked forshape/finitevalues, not independently reexecuted; directnativeforward checks reported byproducer.')
    f=V/'fact_query_panel_audit.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
