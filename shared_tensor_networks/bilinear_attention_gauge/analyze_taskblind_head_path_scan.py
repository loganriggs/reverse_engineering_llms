"""Contrast mean, coverage, and input-energy head selection without labels."""

import json

import numpy as np
from scipy.stats import spearmanr

from flagship_transfer import ART


SEED=20261124
BOOTSTRAPS=10000
TRAIN_BOOTSTRAPS=2000


def greedy(matrix,k):
    covered=np.zeros(len(matrix))
    chosen=[]
    for _ in range(k):
        gains=np.maximum(covered[:,None],np.maximum(matrix,0)).mean(axis=0)-covered.mean()
        gains[chosen]=-np.inf
        index=int(np.argmax(gains))
        chosen.append(index)
        covered=np.maximum(covered,matrix[:,index].clip(min=0))
    return chosen


def run():
    source=json.loads((ART/'taskblind_head_path_scan.json').read_text())
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    input_rank=json.loads((ART/'all_head_unlabeled_edit_scan.json').read_text())['records']
    heads=sorted({(x['layer'],x['head']) for x in source['rows']})
    head_index={h:i for i,h in enumerate(heads)}
    arrays={};types={}
    for panel in ('train','validation'):
        case_ids=sorted({x['case_index'] for x in source['rows'] if x['panel']==panel})
        ids={case_id:i for i,case_id in enumerate(case_ids)}
        values=np.zeros((len(case_ids),len(heads)))
        for row in source['rows']:
            if row['panel']==panel:
                values[ids[row['case_index']],head_index[(row['layer'],row['head'])]]=(
                    row['signed_projection'])
        assert values.shape==(len(source['types'])*source['n_per_type'],36)
        arrays[panel]=values
        types[panel]=np.array([meta['panels'][panel][i]['type'] for i in case_ids])
        assert all(np.sum(types[panel]==kind)==source['n_per_type'] for kind in source['types'])
    train,val=arrays['train'],arrays['validation']
    mean_rank=np.argsort(-train.mean(axis=0)).tolist()
    coverage_rank=greedy(train,8)
    input_order=[head_index[(x['layer'],x['head'])] for x in input_rank]
    rankings={'mean_path':mean_rank,'greedy_coverage':coverage_rank,'input_response_energy':input_order}
    cover_results={}
    for name,indices in rankings.items():
        cover_results[name]={}
        for k in (1,2,3,5,8):
            score=np.maximum(val[:,indices[:k]],0).max(axis=1)
            cover_results[name][str(k)]=dict(
                selected_heads=[list(heads[j]) for j in indices[:k]],
                validation_mean_max_positive_projection=float(np.mean(score)),
                validation_mean_by_edit_type={kind:float(np.mean(score[types['validation']==kind]))
                    for kind in source['types']})
    rng=np.random.default_rng(SEED)
    groups=[np.flatnonzero(types['validation']==kind) for kind in source['types']]
    choices=np.concatenate([rng.choice(g,size=(BOOTSTRAPS,len(g)),replace=True)
                            for g in groups],axis=1)
    top3={name:np.maximum(val[:,indices[:3]],0).max(axis=1)
          for name,indices in rankings.items()}
    bootstrap={}
    for other in ('mean_path','input_response_energy'):
        diff=top3['greedy_coverage']-top3[other]
        distribution=diff[choices].mean(axis=1)
        bootstrap[f'greedy_minus_{other}']=dict(mean=float(np.mean(diff)),
            stratified_case_bootstrap_ci95=np.quantile(distribution,[.025,.975]).tolist())
    train_groups=[np.flatnonzero(types['train']==kind) for kind in source['types']]
    inclusion={str(h):0 for h in heads}
    top3_exact=0
    for _ in range(TRAIN_BOOTSTRAPS):
        selected=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in train_groups])
        fitted=greedy(train[selected],3)
        for j in fitted:inclusion[str(heads[j])]+=1
        top3_exact+=set(fitted)==set(coverage_rank[:3])
    inclusion={name:value/TRAIN_BOOTSTRAPS for name,value in inclusion.items() if value>0}
    heldout={}
    for kind in source['types']:
        subset=train[types['train']!=kind]
        excluded_greedy=greedy(subset,5)
        per_type=val[types['validation']==kind].mean(axis=0)
        oracle=int(np.argmax(per_type))
        heldout[kind]=dict(validation_best_head=list(heads[oracle]),
            validation_best_head_mean_signed_projection=float(per_type[oracle]),
            greedy_top5_excluding_type=[list(heads[j]) for j in excluded_greedy],
            validation_best_head_in_top5=oracle in excluded_greedy,
            validation_best_head_all_type_mean_rank=mean_rank.index(oracle)+1)
    # Independent implementation check against an earlier sequential patcher.
    previous=json.loads((ART/'punctuation_head_finite.json').read_text())
    reference={(r['case_index'],r['head']):r for r in previous['rows']}
    discrepancies=[]
    for row in source['rows']:
        key=(row['case_index'],row['head'])
        if row['panel']=='validation' and row['layer']==13 and key in reference:
            assert reference[key]['kind']==row['type']
            discrepancies.append(abs(row['signed_projection']-
                reference[key]['full_head']['signed_projection']))
    assert len(discrepancies)==324 and max(discrepancies)<1e-5
    train_scores=train.mean(axis=0);val_scores=val.mean(axis=0)
    result=dict(description='Label-free 36-head finite path ranking: mean individual-head signed projection versus greedy max-positive coverage, input-state-energy baseline, stratified validation bootstrap, train resampling, and whole-edit-type exclusions',
        source='taskblind_head_path_scan.json',seed=SEED,
        bootstrap_replicates=BOOTSTRAPS,train_bootstrap_replicates=TRAIN_BOOTSTRAPS,
        head_score_train_validation_spearman=float(spearmanr(train_scores,val_scores).statistic),
        mean_path_top8_train_validation_overlap=len(set(mean_rank[:8])&set(np.argsort(-val_scores)[:8])),
        sequential_vectorized_patch_agreement=dict(matched_cases=len(discrepancies),
            max_abs_signed_projection_error=float(max(discrepancies))),
        rankings={name:[list(heads[j]) for j in indices] for name,indices in rankings.items()},
        coverage_results=cover_results,top3_bootstrap=bootstrap,
        greedy_top3_train_resample_exact_set_rate=top3_exact/TRAIN_BOOTSTRAPS,
        greedy_top3_head_inclusion_frequency=inclusion,
        whole_edit_type_exclusion=heldout)
    (ART/'taskblind_head_path_scan_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print('rank agreement',result['head_score_train_validation_spearman'])
    print('top3', {k:v[:3] for k,v in result['rankings'].items()})
    print('coverage', {k:v['3']['validation_mean_max_positive_projection']
                       for k,v in cover_results.items()})
    print('bootstrap',bootstrap)
    print('top3 resample',result['greedy_top3_train_resample_exact_set_rate'],inclusion)
    print('bracket exclusion',heldout['bracket'])


if __name__=='__main__':run()
