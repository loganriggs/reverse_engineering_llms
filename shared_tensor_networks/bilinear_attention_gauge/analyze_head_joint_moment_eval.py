"""Paired KL and token-family path analysis of the joint moment grid."""

import json

import numpy as np

from flagship_transfer import ART


SEED=20261121
BOOTSTRAPS=10000


def run():
    joint=json.loads((ART/'head_joint_moment_eval.json').read_text())
    baseline=json.loads((ART/'output_aware_head_eval.json').read_text())
    fresh=json.loads((ART/'head_joint_bracket_confirmation.json').read_text())
    rng=np.random.default_rng(SEED)
    summary={}
    for target in ('bracket_closure','subject_verb_agreement'):
        summary[target]={}
        for method in joint['methods']:
            records=[x for x in joint['kl_records'] if x['target']==target and x['method']==method]
            assert len(records)==6
            deltas=[]
            for row in records:
                ref=next(x for x in baseline['kl_records'] if x['target']==target and
                    x['corpus']==row['corpus'] and x['length']==row['length'] and
                    x['method']=='weight')
                delta=np.asarray(row['kl'])-np.asarray(ref['kl'])
                n=len(delta)
                deltas.append(delta[rng.integers(0,n,(BOOTSTRAPS,n))].mean(axis=1))
            joint_boot=np.mean(np.stack(deltas),axis=0)
            path=next(x for x in joint['edit_summary'] if x['target']==target and
                      x['panel']=='all' and x['method']==method)
            summary[target][method]=dict(
                equal_panel_mean_kl=float(np.mean([x['mean_kl'] for x in records])),
                equal_panel_mean_kl_delta_vs_weight=float(joint_boot.mean()),
                kl_delta_vs_weight_ci95=np.quantile(joint_boot,[.025,.975]).tolist(),
                total_top1_flips=int(sum(x['flip_count'] for x in records)),
                native_head_path_relative_mae=path['relative_mae_vs_native'],
                task_pairs_correct=path['both_forced_choice_correct'],
                task_pair_count=path['count'])
    prior=json.loads((ART/'competent_bracket_identities.json').read_text())
    names={(tuple(x['base_ids']),tuple(x['edited_ids'])):x['family']
           for x in prior['rows'] if x['panel']=='validation'}
    old_rows={ (tuple(x['before']),tuple(x['after'])):x for x in baseline['edit_rows']
              if x['target']=='bracket_closure'}
    family_stats={}
    for row in joint['edit_rows']:
        if row['target']!='bracket_closure':continue
        key=(tuple(row['before']),tuple(row['after']))
        old=old_rows[key]
        assert abs(row['native_head_path']-old['native_head_path'])<1e-5
        family=names[key]
        family_stats.setdefault(family,[]).append((
            abs(row['effects']['lambda_0.1']-row['native_head_path'])-
            abs(old['effects']['weight']-old['native_head_path']),
            abs(row['native_head_path'])))
    family_names=sorted(family_stats)
    assert len(family_names)==6
    confirmation={}
    fresh_stats={}
    for row in fresh['rows']:
        fresh_stats.setdefault(row['family'],[]).append((
            abs(row['effects']['lambda_0.1']-row['native_head_path'])-
            abs(row['effects']['weight']-row['native_head_path']),
            abs(row['native_head_path'])))
    for label,stats in (('original_validation',family_stats),
                        ('fresh_expressions',fresh_stats)):
        names=sorted(stats)
        selected=rng.integers(0,len(names),(BOOTSTRAPS,len(names)))
        boot=[]
        for group in selected:
            values=[pair for index in group for pair in stats[names[index]]]
            boot.append(sum(v for v,_ in values)/sum(v for _,v in values))
        all_values=[pair for name in names for pair in stats[name]]
        confirmation[label]=dict(family_count=len(names),pair_count=len(all_values),
            lambda_0_1_minus_weight_absolute_error_over_native_path=float(
                sum(v for v,_ in all_values)/sum(v for _,v in all_values)),
            family_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist())
    result=dict(description='Joint rank-four moment grid: paired context KL bootstrap versus frozen weight-only cut, and cluster-by-token-identity path comparison for bracket lambda 0.1 versus weight',
        seed=SEED,bootstrap_replicates=BOOTSTRAPS,
        caveat='The 21 original validation prompts were inspected when choosing a compromise from the declared grid; the 24 fresh expression prompts are a separate context confirmation using the same six token identities.',
        summary=summary,bracket_path_comparison=confirmation)
    (ART/'head_joint_moment_eval_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for target,methods in summary.items():
        print(target,flush=True)
        for name,item in methods.items():print(name,item,flush=True)
    print('bracket confirmation',confirmation,flush=True)


if __name__=='__main__':run()
