"""Independent labels, groups, logits, cells and shortcut diagnostics for M107."""
import json,itertools
import numpy as np
import torch,tiktoken
from fit_pca_encoder import V,sha


def main():
    torch.set_num_threads(1);pp=V/'keyed_copy_protocol.json';rp=V/'keyed_copy_screen.json';p=json.loads(pp.read_text());r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(pp) and r['source_sha256']==sha(V.parent/'screen_keyed_copy.py');assert sha(r['states_path'])==r['states_sha256'];saved=torch.load(r['states_path'],weights_only=True);meta={x['id']:x for x in p['rows']};rows={x['id']:x for x in r['rows']};assert len(meta)==len(rows)==len(saved)==80;tok=tiktoken.get_encoding('gpt2');maximum=0.
    challenge=[x for x in p['rows'] if x['cohort']=='keyed_challenge'];grid={(x['format'],x['case'],x['a'],x['b'],x['query'],x['order']) for x in challenge};assert grid==set(itertools.product(['semicolon','comma'],[0,1],[0,1],[0,1],[0,1],[0,1]));lookup={(x['format'],x['case'],x['a'],x['b'],x['query'],x['order']):x for x in challenge}
    def check(a,b):
        nonlocal maximum
        assert np.isclose(a,b,rtol=1e-10,atol=1e-10);maximum=max(maximum,abs(a-b))
    for key,m in meta.items():
        assert tok.encode(m['prompt'])==m['token_ids']==saved[key]['ids'][0].tolist();correct=m['correct_index']
        if m['cohort']=='keyed_challenge':
            values=[m['values'][m['a']],m['values'][2+m['b']]];assert values==m['assignments'];pairs=[k+' '+v for k,v in zip(m['keys'],values)];pairs=pairs if m['order']==0 else pairs[::-1];sep='; ' if m['format']=='semicolon' else ', ';assert m['prompt']==sep.join(pairs+pairs+[m['keys'][m['query']]]);assert m['values'][correct]==values[m['query']]
            for value,group in zip(m['values'],m['answer_groups']):
                expected=[]
                for form in set([value,value.capitalize(),value.upper()]):
                    ext=tok.encode(m['prompt']+' '+form)
                    if ext[:-1]==m['token_ids'] and len(ext)==len(m['token_ids'])+1:expected.append(ext[-1])
                assert sorted(set(expected))==group
        z=saved[key]['logits'].double().numpy();lse=lambda a:float(a.max()+np.log(np.exp(a-a.max()).sum()));scores=np.array([lse(z[group]) for group in m['answer_groups']]);pred=int(scores.argmax());row=rows[key];assert pred==row['predicted_index'] and row['choice_correct']==(pred==correct);assert row['vocab_top1']==(int(z.argmax()) in m['answer_groups'][correct]);check(float(scores[correct]-max(s for j,s in enumerate(scores) if j!=correct)),row['margin_over_best_wrong'])
    gate=p['gate']
    for cell in r['cells']:
        rr=[x for x in r['rows'] if x.get('format')==cell['format'] and x['query']==cell['query'] and x['order']==cell['order']];assert len(rr)==cell['n']==8;acc=sum(x['choice_correct'] for x in rr)/8;top=sum(x['vocab_top1'] for x in rr)/8;check(acc,cell['choice_accuracy']);check(top,cell['vocab_top1']);assert cell['passed']==(acc>=gate['min_four_way_accuracy'] and top>=gate['min_vocab_top1'])
    diagnostics=[]
    for fmt in ['semicolon','comma']:
        rr=[x for x in r['rows'] if x.get('format')==fmt];summary=next(x for x in r['summaries'] if x['format']==fmt);check(sum(x['choice_correct'] for x in rr)/32,summary['choice_accuracy']);check(sum(x['vocab_top1'] for x in rr)/32,summary['vocab_top1']);assert summary['passed']==all(x['passed'] for x in r['cells'] if x['format']==fmt)
        counts={k:0 for k in ['queried','other_key','unmentioned','first_value','last_value']};querychanges=0;queryboth=0
        for x in rr:
            m=meta[x['id']];av=m['a'];bv=2+m['b'];pred=x['predicted_index'];counts['queried']+=pred==m['correct_index'];counts['other_key']+=pred==(bv if m['query']==0 else av);counts['unmentioned']+=pred not in [av,bv];counts['first_value']+=pred==(av if m['order']==0 else bv);counts['last_value']+=pred==(bv if m['order']==0 else av)
            if m['query']==0:
                other=lookup[fmt,m['case'],m['a'],m['b'],1,m['order']];oy=rows[other['id']];querychanges+=pred!=oy['predicted_index'];queryboth+=x['choice_correct'] and oy['choice_correct']
        assert counts['queried']+counts['other_key']+counts['unmentioned']==32;diagnostics.append(dict(format=fmt,counts=counts,query_switch_pairs=16,query_choice_changes=querychanges,query_both_correct=queryboth))
    historical=[x for x in r['rows'] if x['cohort']=='historical_replay'];assert len(historical)==r['historical_replays']==16 and r['forward_count']==80
    out=dict(status='passed',source_sha256=sha(__file__),result_sha256=sha(rp),protocol_sha256=sha(pp),prompts_checked=80,cell_gates=8,maximum_margin_error=maximum,historical_choice_accuracy=sum(x['choice_correct'] for x in historical)/16,historical_top1=sum(x['vocab_top1'] for x in historical)/16,diagnostics=diagnostics,scope='Independentallprompt labels/tokenizations/logits/groups andgates. Descriptive candidate-choice shortcut accounting added after outcomes, no heuristic fitted. Historicalbaselineexactreplayverifiedbyproducer; no independentTransformer. Two-key contexts arelonger, sofailuredoesnot uniquelyidentify recency orkey-routing failure.')
    f=V/'keyed_copy_audit.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
