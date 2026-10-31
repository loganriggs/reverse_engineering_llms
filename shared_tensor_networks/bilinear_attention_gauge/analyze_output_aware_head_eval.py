"""Paired KL bootstrap and cross-metric table for output-aware head cuts."""

import json

import numpy as np

from flagship_transfer import ART


SEED=20261119
BOOTSTRAPS=10000


def run():
    source=json.loads((ART/'output_aware_head_eval.json').read_text())
    rng=np.random.default_rng(SEED)
    panels=[];aggregate={}
    for target in ('bracket_closure','subject_verb_agreement'):
        boots={method:[] for method in source['methods'] if method!='weight'}
        for length in (32,64):
            for corpus in ('fineweb_test','wikitext103','tinystories'):
                rows={x['method']:x for x in source['kl_records'] if
                      x['target']==target and x['length']==length and x['corpus']==corpus}
                assert set(rows)==set(source['methods'])
                n=len(rows['weight']['kl'])
                assert all(len(x['kl'])==n for x in rows.values())
                idx=rng.integers(0,n,size=(BOOTSTRAPS,n))
                differences={}
                for method in boots:
                    delta=np.asarray(rows[method]['kl'])-np.asarray(rows['weight']['kl'])
                    sample=delta[idx].mean(axis=1)
                    boots[method].append(sample)
                    differences[method]=dict(mean=float(delta.mean()),
                        ci95=np.quantile(sample,[.025,.975]).tolist(),
                        fraction_contexts_method_lower=float(np.mean(delta<0)))
                panels.append(dict(target=target,corpus=corpus,length=length,
                    count=n,mean_kl={k:v['mean_kl'] for k,v in rows.items()},
                    flips={k:v['flip_count'] for k,v in rows.items()},
                    method_minus_weight_kl=differences))
        aggregate[target]={}
        for method in source['methods']:
            method_rows=[x for x in source['kl_records'] if x['target']==target and x['method']==method]
            summary=next(x for x in source['edit_summary'] if x['target']==target
                         and x['panel']=='all' and x['method']==method)
            entry=dict(equal_panel_mean_kl=float(np.mean([x['mean_kl'] for x in method_rows])),
                total_top1_flips=int(sum(x['flip_count'] for x in method_rows)),
                edit_path_relative_mae=summary['relative_mae_vs_native'],
                edit_forced_choice_correct=summary['both_forced_choice_correct'],
                edit_case_count=summary['count'])
            if method!='weight':
                joint=np.mean(np.stack(boots[method]),axis=0)
                entry['equal_panel_mean_kl_difference_vs_weight']=float(joint.mean())
                entry['difference_ci95']=np.quantile(joint,[.025,.975]).tolist()
            aggregate[target][method]=entry
    result=dict(description='Paired context bootstrap of new data-aware rank-four cuts against frozen weight cut; output reduction evaluated separately on native head edit paths',
        source='output_aware_head_eval.json',seed=SEED,bootstrap_replicates=BOOTSTRAPS,
        caveat='Panel bootstrap treats sampled windows as independent and does not account for source-document overlap across context lengths.',
        panels=panels,aggregate=aggregate)
    (ART/'output_aware_head_eval_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for target,methods in aggregate.items():
        print(target,flush=True)
        for name,row in methods.items():print(name,row,flush=True)


if __name__=='__main__':run()
