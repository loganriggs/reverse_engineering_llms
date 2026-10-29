"""M107 discriminate keyed retrieval from last-value copying, validation only."""
import json,itertools
import tiktoken
from fit_pca_encoder import V,sha


def main():
    tok=tiktoken.get_encoding('gpt2');oldpath=V/'whole_task_native_protocol.json';old=json.loads(oldpath.read_text());rows=[]
    for pair in old['pairs']:
        if pair['family']!='sequence_copy':continue
        for ep in [0,1]:rows.append(dict(id=pair['id']+f'|{ep}',cohort='historical_replay',token_ids=pair['token_ids'][ep],prompt=tok.decode(pair['token_ids'][ep]),answer_groups=pair['answer_groups'][ep],correct_index=pair['variants'][ep]))
    specs=[dict(keys=['white','orange'],values=['red','yellow','blue','black']),dict(keys=['purple','red'],values=['green','white','yellow','orange'])]
    for case,spec in enumerate(specs):
        for fmt,sep in [('semicolon','; '),('comma',', ')]:
            for a,b,q,order in itertools.product([0,1],repeat=4):
                values=[spec['values'][a],spec['values'][2+b]];pairs=[k+' '+v for k,v in zip(spec['keys'],values)];pairs=pairs if order==0 else pairs[::-1];prompt=sep.join(pairs+pairs+[spec['keys'][q]]);ids=tok.encode(prompt);groups=[]
                for value in spec['values']:
                    group=[]
                    for form in dict.fromkeys([value,value.capitalize(),value.upper()]):
                        ext=tok.encode(prompt+' '+form)
                        if ext[:-1]==ids and len(ext)==len(ids)+1:group.append(ext[-1])
                    groups.append(sorted(set(group)))
                assert all(groups) and len(set(sum(groups,[])))==sum(map(len,groups))
                rows.append(dict(id=f'm107:{fmt}:{case}:{a}{b}{q}{order}',cohort='keyed_challenge',format=fmt,case=case,a=a,b=b,query=q,order=order,keys=spec['keys'],values=spec['values'],assignments=values,prompt=prompt,token_ids=ids,answer_groups=groups,correct_index=a if q==0 else 2+b))
    assert len(rows)==80 and len({x['prompt'] for x in rows})==80
    oldtrain=json.loads((V/'training_causal_coverage_protocol.json').read_text());trainids={tuple(ids) for p in oldtrain['pairs'] for ids in p['token_ids']};assert all(tuple(x['token_ids']) not in trainids for x in rows)
    out=dict(status='frozen_before_new_outcomes',source_sha256=sha(__file__),historical_protocol_sha256=sha(oldpath),training_protocol_sha256=sha(V/'training_causal_coverage_protocol.json'),rows=rows,specs=specs,gate=dict(unit='punctuation format',min_n_per_query_order_cell=8,min_four_way_accuracy=.75,min_vocab_top1=.5,all_four_cells_required=True),scope='Existingone-key sequencecopy canbesolvedbylastvaluecopy. Fixed2key repeatedassociations withquery/order/factvalues balanced distinguishthatshortcut. All64newprompts plus16exacthistoricalreplays. Samecontentwordsandpunctuationfamily, longercontexts; notalengthmatched causalcontrast. Whole sequencecopy family remains excludedfromanyfeaturizer training. Notrepairing M106 naturalfactprompt outcomes; separate test ofa loophole inpreviouspositivecopy evidence.',stopping='One fixedscreen; keepallfailures, no variants/promptreplacement/gatethresholdchange. Ifpass, freeze internalcausalrouting/selectivity test beforeanyfeaturizer outcome. If fail, restrict priorcopyclaim tounkeyedrepetition ratherthan inferkeyedretrieval. No newtaskdiscoveryclaim.')
    f=V/'keyed_copy_protocol.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('FROZEN',len(rows),'16replay+64challenge')
if __name__=='__main__':main()
