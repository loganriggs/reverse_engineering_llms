"""Summarize held-out directional pair and full-model key-edit tests."""

import json

import numpy as np

from flagship_transfer import ART


SOURCE='all_head_shared_key_directional_reversal_n8_s111-117.json'
PAIR='all_head_shared_key_directional_pair_probe_n48.json'
METHODS=('forward_c16','reverse_c165','random_c94','random_c106')
FORMS=('space_word','punctuation')


def pooled(rows,numerator,denominator=None):
    n=sum(r[numerator] for r in rows)
    return n/sum(r[denominator] for r in rows) if denominator else n/len(rows)


def run():
    panel=json.loads((ART/SOURCE).read_text())
    pair=json.loads((ART/PAIR).read_text())
    records=panel['key_edit_responses']
    assert len(panel['fidelity'])==48 and len(records)==48*len(METHODS)*2
    assert len(pair['records'])==48*len(METHODS)
    keys={(r['corpus'],r['seed'],r['document_id']) for r in panel['fidelity']}
    assert len(keys)==48
    result={'source':SOURCE,'pair_source':PAIR,'windows':len(keys),
            'per_method':{},'by_corpus':{}}
    for method in METHODS:
        result['per_method'][method]={}
        for form in FORMS:
            a=[r for r in records if r['method']==method and
               r['query_form']==form]
            result['per_method'][method][form]={
                'logit_response_fraction':pooled(a,'component_logit_response_energy',
                                                  'baseline_logit_response_energy'),
                'head_response_fraction':pooled(a,'component_head_response_energy',
                                                 'baseline_head_response_energy'),
                'mean_logit_component_energy':pooled(a,'component_logit_response_energy'),
                'mean_head_component_energy':pooled(a,'component_head_response_energy')}
    for corpus in sorted({r['corpus'] for r in records}):
        result['by_corpus'][corpus]={}
        for method in METHODS[:2]:
            result['by_corpus'][corpus][method]={
                form:pooled([r for r in records if r['corpus']==corpus and
                             r['method']==method and r['query_form']==form],
                            'component_logit_response_energy',
                            'baseline_logit_response_energy')
                for form in FORMS}
    pair_rows=pair['records']
    result['selected_pair_energy']={}
    for method in METHODS:
        a=[r for r in pair_rows if r['method']==method]
        result['selected_pair_energy'][method]={
            field:pooled(a,field) for field in (
                'word_query_punct_key_pair_energy',
                'punct_query_word_key_pair_energy',
                'word_query_key_edit_pair_energy',
                'punct_query_key_edit_pair_energy')}
    fidelity=panel['fidelity']
    result['native_to_reduced']={
        'interaction_relative_squared_error':pooled(
            fidelity,'interaction_error_energy','native_interaction_energy'),
        'key_edit_relative_squared_error':{
            form:(sum(r['native_to_reduced_key_edit_response'][form]['error_energy']
                      for r in fidelity)/
                  sum(r['native_to_reduced_key_edit_response'][form]['native_energy']
                      for r in fidelity)) for form in FORMS},
        'mean_query_kl_by_condition':{
            c:float(np.mean([r['native_to_reduced_query_kl'][c] for r in fidelity]))
            for c in panel['conditions']}}
    result['identity_projection_max_abs_logit']=panel['identity_projection_max_abs_logit']

    # Resample windows independently within corpus, retaining paired methods.
    grouped={c:sorted(k for k in keys if k[0]==c) for c in sorted({k[0] for k in keys})}
    lookup={(r['corpus'],r['seed'],r['document_id'],r['method'],r['query_form']):r
            for r in records}
    rng=np.random.default_rng(20261031)
    draws=[]
    for _ in range(10000):
        sample=[k for ks in grouped.values() for k in
                (ks[i] for i in rng.integers(0,len(ks),len(ks)))]
        values={}
        for method in METHODS[:2]:
            for form in FORMS:
                a=[lookup[(*k,method,form)] for k in sample]
                values[method,form]=pooled(a,'component_logit_response_energy',
                                           'baseline_logit_response_energy')
        draws.append((values['forward_c16','space_word']-
                      values['reverse_c165','space_word'])-
                     (values['forward_c16','punctuation']-
                      values['reverse_c165','punctuation']))
    observed=(result['per_method']['forward_c16']['space_word']['logit_response_fraction']-
              result['per_method']['reverse_c165']['space_word']['logit_response_fraction']-
              result['per_method']['forward_c16']['punctuation']['logit_response_fraction']+
              result['per_method']['reverse_c165']['punctuation']['logit_response_fraction'])
    result['full_model_double_dissociation']={
        'contrast':observed,'bootstrap_95_interval':np.quantile(draws,[.025,.975]).tolist(),
        'definition':'(forward minus reverse key-edit response fraction under space-word query) minus (forward minus reverse under punctuation query)'}
    (ART/'all_head_shared_key_directional_summary_n48.json').write_text(
        json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':run()
