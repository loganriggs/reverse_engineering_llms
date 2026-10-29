"""M106 two independently varied facts, queried entity and order; no outcomes."""
import json,itertools
import tiktoken
from fit_pca_encoder import V,sha

SPECS={
 'color_binding':dict(fact="{name}'s favorite color is {value}.",completion="{name}'s favorite color is",question="What is {name}'s favorite color?",answer_prefix='Answer:',values=[['red','green','blue','yellow'],['black','white','brown','orange']]),
 'object_binding':dict(fact='{name} has a {value}.',completion='{name} has a',question='What does {name} have?',answer_prefix='Answer: a',values=[['book','key','cup','pen'],['ticket','coin','bag','hat']]),
 'city_binding':dict(fact='{name} lives in {value}.',completion='{name} lives in',question='Where does {name} live?',answer_prefix='Answer:',values=[['Paris','Rome','London','Berlin'],['Boston','Denver','Madrid','Tokyo']])}


def main():
    tok=tiktoken.get_encoding('gpt2');rows=[];old=set();oldhash={}
    for stem in ['training_causal_coverage_protocol','source_holdout_native_protocol','whole_task_native_protocol']:
        path=V/(stem+'.json');p=json.loads(path.read_text());oldhash[path.name]=sha(path)
        for pair in p['pairs']:
            for ids in pair['token_ids']:old.add(tuple(ids))
    for domain,spec in SPECS.items():
        for fmt in ['completion','question_answer']:
            for case in [0,1]:
                names=[['Alice','Bob'],['Carol','Dave']][case];values=spec['values'][case]
                for a,b,q,order in itertools.product([0,1],repeat=4):
                    assignments=[values[a],values[2+b]];facts=[spec['fact'].format(name=n,value=v) for n,v in zip(names,assignments)];facts=facts if order==0 else facts[::-1];prompt=' '.join(facts)
                    prompt+=' '+spec['completion'].format(name=names[q]) if fmt=='completion' else '\nQuestion: '+spec['question'].format(name=names[q])+'\n'+spec['answer_prefix']
                    ids=tok.encode(prompt);answers=[];valid=True
                    for value in values:
                        ext=tok.encode(prompt+' '+value);good=ext[:-1]==ids and len(ext)==len(ids)+1;valid&=good;answers.append(ext[-1] if good else None)
                    valid&=len(set(answers))==4
                    role='potential_training' if case==0 and fmt=='completion' and a==b else 'reserved_combination_or_context'
                    rows.append(dict(id=f'{domain}:{fmt}:{case}:{a}{b}{q}{order}',domain=domain,format=fmt,case=case,a=a,b=b,query=q,order=order,names=names,values=values,assignments=assignments,prompt=prompt,token_ids=ids,answer_token_ids=answers,correct_index=a if q==0 else 2+b,valid=bool(valid),data_role=role,exact_historical_prompt_overlap=tuple(ids) in old))
    assert len(rows)==192 and len({x['prompt'] for x in rows})==192
    lookup={(x['domain'],x['format'],x['case'],x['a'],x['b'],x['query'],x['order']):x for x in rows};edges=[]
    for row in rows:
        for factor in ['a','b','query','order']:
            if row[factor]!=0:continue
            key={k:row[k] for k in ['domain','format','case','a','b','query','order']};key[factor]=1;other=lookup[tuple(key.values())];equal=len(row['token_ids'])==len(other['token_ids']);changed=[i for i,(a,b) in enumerate(zip(row['token_ids'],other['token_ids'])) if a!=b] if equal else None
            edges.append(dict(factor=factor,recipient_id=row['id'],donor_id=other['id'],same_token_length=equal,changed_positions=changed,answer_should_change=factor=='query' or factor=='a' and row['query']==0 or factor=='b' and row['query']==1))
    assert len(edges)==384
    protocol=dict(status='frozen_before_model_outcomes',source_sha256=sha(__file__),tokenizer='tiktoken gpt2',prior_protocol_hashes=oldhash,specs=SPECS,rows=rows,edges=edges,competence_gate=dict(unit='domain x format',cells='each queried_entity x fact_order cell separately',min_valid_examples_per_cell=8,min_four_way_accuracy=.75,min_exact_vocab_top1=.50,require_all_four_cells=True),split='Potential training only lexicalcase0 completion-format diagonal factcombinations a=b; allcrosscombinations, case1 andQAformat reserved fromanylaterfeaturizer fit. Native competence is evaluated throughout beforedeciding usability. No new featurefit now.',stopping='One frozen192prompt screen, no promptreplacement or thresholdretuning afterfailures. Allcohorts/cells retained. Do notcall competence a mechanism; next causal intervention onlyif competencegate passes.',scope='Controlled explicit factretrieval under2independent binaryfacts, queryswitch andorder. Colors/objects/cities provide3contentdomains butshare a retrieval operation, not3newtasks. Different fromearlier usual-object-color lookup andentity giver/recipient completion. Familiar componentoperations; no new-task discovery or pretraining-exclusion claim. Canonical single-token answers, notexhaustive semanticequivalence.')
    dest=V/'fact_query_panel_protocol.json';assert not dest.exists();dest.write_text(json.dumps(protocol,indent=2)+'\n');print('FROZEN',len(rows),'PROMPTS',sum(x['valid'] for x in rows),'VALID',sum(x['exact_historical_prompt_overlap'] for x in rows),'EXACT_OLD_OVERLAPS',len(edges),'EDGES',sum(x['same_token_length'] for x in edges),'ALIGNED')
if __name__=='__main__':main()
