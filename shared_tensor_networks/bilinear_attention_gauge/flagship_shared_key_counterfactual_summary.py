"""Paired, stratified uncertainty analysis of shared-key input-edit mediation."""

import json
from collections import defaultdict

import numpy as np

from flagship_transfer import ART


PANELS=(
 'all_head_shared_key_counterfactual_causality_n8_s91-97.json',
 'all_head_shared_key_counterfactual_causality_n8_s101-107_punct25-26_word286-257.json',
)
PAIRS=(
 ('response_0_c16_matched','random_matched_to_response_0_c94'),
 ('response_0_c16_matched','same_gauge_response_0_c165'),
 ('direct_l1_native_0_c3_matched','random_matched_to_direct_l1_native_0_c106'),
 ('direct_l1_native_0_c3_matched','same_gauge_direct_l1_native_0_c1'),
)
FIDELITY='all_head_shared_key_input_edit_fidelity_n96.json'
SAME_GAUGE='all_head_shared_key_same_gauge_controls_n96.json'


def statistic(rows,candidate,control):
    p=sum(row[candidate]['punctuation_logit_response_energy'] for row in rows)
    w=sum(row[candidate]['word_logit_response_energy'] for row in rows)
    assert min(p,w)>0
    def selective(name):
        return (sum(row[name]['punctuation_mediated_logit_energy']
                    for row in rows)/p-
                sum(row[name]['word_mediated_logit_energy']
                    for row in rows)/w)
    head_candidate=sum(row[candidate]['base_head_deletion_energy']
                       for row in rows)
    head_control=sum(row[control]['base_head_deletion_energy']
                     for row in rows)
    return dict(candidate_specificity=selective(candidate),
                control_specificity=selective(control),
                specificity_difference=selective(candidate)-selective(control),
                test_head_dose_ratio=head_candidate/head_control,
                punctuation_mediated_fraction_candidate=sum(
                    row[candidate]['punctuation_mediated_logit_energy']
                    for row in rows)/p,
                word_mediated_fraction_candidate=sum(
                    row[candidate]['word_mediated_logit_energy']
                    for row in rows)/w)


def run(bootstrap=10000):
    mapping={}
    for panel in PANELS:
        data=json.loads((ART/panel).read_text())
        for row in data['records']:
            key=(panel,row['corpus'],row['seed'],row['document_id'])
            mapping.setdefault(key,{})[row['method']]=row
    same=json.loads((ART/SAME_GAUGE).read_text())
    assert len(same['records'])==192
    for row in same['records']:
        key=(row['source'],row['corpus'],row['seed'],row['document_id'])
        assert key in mapping
        mapping[key][row['method']]=row
    assert len(mapping)==96
    seen={(k[1],k[3]) for k in mapping}
    assert len(seen)==96,'document overlap between panels'
    for value in mapping.values():
        assert len(value)==8
    rng=np.random.default_rng(1117)
    output=[]
    for panel in (*PANELS,'combined'):
        subset=[key for key in mapping if panel=='combined' or key[0]==panel]
        keys_by_stratum=defaultdict(list)
        for key in subset:keys_by_stratum[(key[0],key[1])].append(key)
        rows=[mapping[key] for key in subset]
        for candidate,control in PAIRS:
            point=statistic(rows,candidate,control)
            boot=np.empty(bootstrap)
            for sample in range(bootstrap):
                chosen=[]
                for group in keys_by_stratum.values():
                    ix=rng.integers(0,len(group),size=len(group))
                    chosen.extend(mapping[group[i]] for i in ix)
                boot[sample]=statistic(chosen,candidate,control)[
                    'specificity_difference']
            output.append(dict(panel=panel,candidate=candidate,control=control,
                n_windows=len(rows),**point,
                bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
                bootstrap_fraction_nonpositive=float(np.mean(boot<=0))))
    fidelity=json.loads((ART/FIDELITY).read_text())
    assert len(fidelity['records'])==96
    fidelity_summary=[]
    for panel in (*PANELS,'combined'):
        rows=[r for r in fidelity['records']
              if panel=='combined' or r['source']==panel]
        fidelity_summary.append(dict(panel=panel,n_windows=len(rows),
            punctuation_response_relative_squared_error=sum(
                r['punctuation_response_error_energy'] for r in rows)/sum(
                r['native_punctuation_response_energy'] for r in rows),
            word_response_relative_squared_error=sum(
                r['word_response_error_energy'] for r in rows)/sum(
                r['native_word_response_energy'] for r in rows),
            mean_base_query_kl=float(np.mean([r['base_query_kl']
                                              for r in rows]))))
    result=dict(panels=list(PANELS),fidelity_artifact=FIDELITY,
        same_gauge_control_artifact=SAME_GAUGE,
        specificity_definition='pooled punctuation-mediated logit energy / pooled punctuation-response energy minus analogous word-control fraction',
        bootstrap='resample windows with replacement within each panel and corpus, 10000 draws, percentile interval',
        records=output,native_fidelity=fidelity_summary,
        scope='exploratory validation-only label selection and held-out full-model counterfactuals; component effects are in the rank-256 explanatory circuit')
    (ART/'all_head_shared_key_counterfactual_summary.json').write_text(
        json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()
