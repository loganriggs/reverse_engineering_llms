"""Paired-document summary of finite QK1-key coordinate deletions."""

import argparse
import json

import numpy as np

from flagship_transfer import ART


def run(source,bootstrap=10000):
    panel=json.loads((ART/source).read_text())
    methods=list(panel['detail'])
    corpora={}
    rng=np.random.default_rng(31917)
    for corpus in ('fineweb_test','wikitext103','tinystories'):
        selected=[r for r in panel['records'] if r['corpus']==corpus]
        by_doc={}
        for r in selected:
            by_doc.setdefault(r['document_id'],{}).setdefault(r['method'],{})[
                r['key_category']]=r
        assert len(by_doc)==panel['n_docs_per_corpus']
        assert all(set(d)==set(methods) for d in by_doc.values())
        assert all(set(m)=={'punctuation','space_word'}
                   for d in by_doc.values() for m in d.values())
        docs=list(by_doc.values());n=len(docs)
        effect={method:{category:np.array([
            doc[method][category]['logprob_change_sq'] for doc in docs])
            for category in ('punctuation','space_word')}
            for method in methods}
        full_effect=None
        if 'full_logit_change_sq' in selected[0]:
            full_effect={method:{category:np.array([
                doc[method][category]['full_logit_change_sq'] for doc in docs])
                for category in ('punctuation','space_word')}
                for method in methods}
        dose={method:{category:np.array([
            doc[method][category]['head_output_change_sq'] for doc in docs])
            for category in ('punctuation','space_word')}
            for method in methods}
        full_dose=None
        if 'head_output_full_sequence_change_sq' in selected[0]:
            full_dose={method:{category:np.array([
                doc[method][category]['head_output_full_sequence_change_sq']
                for doc in docs])
                for category in ('punctuation','space_word')}
                for method in methods}
        baseline_error=np.array([abs(doc[methods[0]]['punctuation']['baseline_logprob']-
                     doc[methods[0]]['punctuation']['native_logprob'])
                     for doc in docs])
        idx=rng.integers(0,n,size=(bootstrap,n))
        def fractions(array):
            p=array['punctuation'][idx].sum(1)
            w=array['space_word'][idx].sum(1)
            return p/np.maximum(p+w,1e-30)
        summaries={}
        for method in methods:
            p=float(effect[method]['punctuation'].sum())
            w=float(effect[method]['space_word'].sum())
            hp=float(dose[method]['punctuation'].sum())
            hw=float(dose[method]['space_word'].sum())
            summaries[method]=dict(
                punctuation_effect_fraction=p/max(p+w,1e-30),
                punctuation_to_word_effect_ratio=p/max(w,1e-30),
                punctuation_head_dose_fraction=hp/max(hp+hw,1e-30),
                punctuation_to_word_head_dose_ratio=hp/max(hw,1e-30),
                pooled_logprob_effect_sq_punctuation=p,
                pooled_logprob_effect_sq_space_word=w,
                pooled_head_dose_sq_punctuation=hp,
                pooled_head_dose_sq_space_word=hw)
            if full_dose is not None:
                shp=float(full_dose[method]['punctuation'].sum())
                shw=float(full_dose[method]['space_word'].sum())
                summaries[method]['punctuation_full_sequence_head_dose_fraction']=\
                    shp/max(shp+shw,1e-30)
                summaries[method]['pooled_full_sequence_head_dose_sq_punctuation']=shp
                summaries[method]['pooled_full_sequence_head_dose_sq_space_word']=shw
            if full_effect is not None:
                fp=float(full_effect[method]['punctuation'].sum())
                fw=float(full_effect[method]['space_word'].sum())
                summaries[method]['punctuation_full_logit_effect_fraction']=\
                    fp/max(fp+fw,1e-30)
                summaries[method]['pooled_full_logit_effect_sq_punctuation']=fp
                summaries[method]['pooled_full_logit_effect_sq_space_word']=fw
        paired={}
        controls=['same_gauge_other_label','random_gauge_matched_energy']
        if 'pair_response_punctuation_axis_matched' in methods:
            controls.append('pair_response_punctuation_axis_matched')
        for name in ('punctuation_axis_1','punctuation_axis_2'):
            for control in controls:
                diff=fractions(effect[name])-fractions(effect[control])
                paired[f'{name}_minus_{control}']=dict(
                    punctuation_effect_fraction_difference=float(
                        summaries[name]['punctuation_effect_fraction']-
                        summaries[control]['punctuation_effect_fraction']),
                    bootstrap_95_percentile=list(map(float,np.quantile(diff,[.025,.975]))),
                    candidate_punctuation_effect_larger_documents=int(np.sum(
                        effect[name]['punctuation']>effect[name]['space_word'])),
                    control_punctuation_effect_larger_documents=int(np.sum(
                        effect[control]['punctuation']>effect[control]['space_word'])))
                if full_effect is not None:
                    diff_full=fractions(full_effect[name])-fractions(full_effect[control])
                    paired[f'{name}_minus_{control}']['full_logit_fraction_difference']=float(
                        summaries[name]['punctuation_full_logit_effect_fraction']-
                        summaries[control]['punctuation_full_logit_effect_fraction'])
                    paired[f'{name}_minus_{control}']['full_logit_bootstrap_95_percentile']=list(
                        map(float,np.quantile(diff_full,[.025,.975])))
        corpora[corpus]=dict(n_documents=n,
            mean_abs_baseline_native_logprob_difference=float(baseline_error.mean()),
            mean_pair_position_gap=float(np.mean([d[methods[0]]['punctuation']['position_gap']
                                                  for d in docs])),
            methods=summaries,paired=paired)
    output=dict(source=source,bootstrap=bootstrap,
        bootstrap_unit='document with paired punctuation and space-word key positions',
        interpretation='conditional finite deletion effects in fixed rank-256 reduced model; not input-token counterfactual',
        corpora=corpora)
    (ART/source.replace('.json','_analysis.json')).write_text(
        json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('--bootstrap',type=int,default=10000)
    a=p.parse_args()
    run(a.source,a.bootstrap)
