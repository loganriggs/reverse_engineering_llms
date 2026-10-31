"""Paired document and bootstrap summary of 546M QK2 response fidelity."""

import json
import numpy as np

from flagship_transfer import ART


FILES=('all_head_shared_key_branch_response_n8_s891_l128_d50.json',
       'all_head_shared_key_branch_response_n16_s892_l128_d50.json')


def run(files=FILES,bootstrap=10000,output_name='all_head_shared_key_branch_response_summary_n24_d50.json'):
    sources=[json.loads((ART/name).read_text()) for name in files]
    edit_families={source.get('edit_family','constant_one') for source in sources}
    assert len(edit_families)==1, edit_families
    for source in sources:
        assert source['length']==128 and source['dose']==.05
        assert source['ordinary_identity_max_abs_logit']<1e-3
        assert source['edited_identity_max_abs_logit']<1e-3
    records=[]
    for source in sources:
        for row in source['records']:
            records.append(dict(row,panel_seed=source['seed']))
    methods=sources[0]['methods']
    result={}
    rng=np.random.default_rng(20261031)
    for corpus in ('fineweb_test','wikitext103','tinystories'):
        selected=[row for row in records if row['corpus']==corpus]
        by_doc={}
        for row in selected:
            key=(row['panel_seed'],row['document_id'],row['start'])
            by_doc.setdefault(key,{})[row['method']]=row
        assert all(set(value)==set(methods) for value in by_doc.values())
        assert len({(key[1],key[2]) for key in by_doc})==len(by_doc)
        docs=list(by_doc.values())
        denominator=np.array([doc[methods[0]]['native_response_energy'] for doc in docs])
        assert np.all(denominator>0)
        sse={method:np.array([doc[method]['response_sse'] for doc in docs])
             for method in methods}
        summaries={method:dict(
            pooled_response_relative_mse=float(sse[method].sum()/denominator.sum()),
            median_document_response_relative_mse=float(np.median(
                [doc[method]['response_relative_mse'] for doc in docs])),
            mean_ordinary_kl=float(np.mean([doc[method]['ordinary_kl'] for doc in docs])),
            mean_edited_kl=float(np.mean([doc[method]['edited_kl'] for doc in docs])))
            for method in methods}
        paired={}
        for challenger in ('function_aware_256','usage_key_256','uniform_key_256'):
            diff=sse['branch_aware_256']-sse[challenger]
            indexes=rng.integers(0,len(docs),size=(bootstrap,len(docs)))
            distribution=diff[indexes].sum(axis=1)/denominator[indexes].sum(axis=1)
            paired[challenger]=dict(
                branch_minus_challenger_pooled_mse=float(diff.sum()/denominator.sum()),
                bootstrap_95_percentile=list(map(float,np.quantile(distribution,[.025,.975]))),
                branch_wins_documents=int(np.sum(diff<0)),
                n_documents=len(docs))
        result[corpus]=dict(n_documents=len(docs),
            response_energy_effective_n=float(denominator.sum()**2/np.sum(denominator**2)),
            largest_document_energy_share=float(denominator.max()/denominator.sum()),
            methods=summaries,paired=paired)
    output=dict(source_files=list(files),edit_family=next(iter(edit_families)),bootstrap=bootstrap,
        bootstrap_unit='document within each corpus',
        metric='full-model edited-minus-unedited full-vocabulary logit response relative SSE',
        corpora=result)
    (ART/output_name).write_text(
        json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--files',nargs='+',default=FILES)
    parser.add_argument('--output-name',default='all_head_shared_key_branch_response_summary_n24_d50.json')
    args=parser.parse_args()
    run(files=args.files,output_name=args.output_name)
