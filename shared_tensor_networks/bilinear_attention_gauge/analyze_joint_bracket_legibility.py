"""Summarize fixed-cut group fidelity and matched random gauges."""

import json

import numpy as np

from flagship_transfer import ART


SEED=20261123
BOOTSTRAPS=10000


def run():
    source=json.loads((ART/'joint_bracket_legibility.json').read_text())
    rows=source['rows']
    families=sorted({x['family'] for x in rows})
    assert len(rows)==24 and len(families)==6
    rng=np.random.default_rng(SEED)
    choice=rng.integers(0,len(families),(BOOTSTRAPS,len(families)))
    methods=source['methods']
    summary={}
    for name in methods:
        summary[name]={}
        for rank in ('rank1','rank2'):
            parts=[]
            for family in families:
                selected=[x for x in rows if x['family']==family]
                parts.append((sum(abs(x['effects'][name][rank]-x['joint_rank4_path'])
                                  for x in selected),
                              sum(abs(x['joint_rank4_path']) for x in selected)))
            parts=np.asarray(parts)
            boot=parts[choice].sum(axis=1)
            values=boot[:,0]/boot[:,1]
            summary[name][rank]=dict(
                relative_mae_vs_joint_rank4=source['path_summary'][name][rank][
                    'relative_mae_vs_joint_rank4'],
                token_family_bootstrap_ci95=np.quantile(values,[.025,.975]).tolist(),
                relative_mae_vs_native=source['path_summary'][name][rank][
                    'relative_mae_vs_native'])
    random={}
    for rank in ('rank1','rank2'):
        values=np.array([record[rank]['relative_mae_vs_joint_rank4']
            for name,record in source['path_summary'].items() if name.startswith('random_')])
        assert len(values)==source['random_controls']
        random[rank]=dict(count=len(values),median=float(np.median(values)),
            min=float(np.min(values)),max=float(np.max(values)),
            random_no_better_than_raw_pca=int(sum(values<=summary['raw_pca'][rank][
                'relative_mae_vs_joint_rank4'])),
            conservative_empirical_rank_p=(1+int(sum(values<=summary['raw_pca'][rank][
                'relative_mae_vs_joint_rank4'])))/(1+len(values)))
    result=dict(description='Token-identity-family bootstrap of finite response group fidelity inside exact joint rank-four bracket gauge, with 16 matched random rotation controls',
        source='joint_bracket_legibility.json',seed=SEED,
        bootstrap_replicates=BOOTSTRAPS,methods=summary,random=random,
        train_validation_raw_pca_axis_cosine=source['train_validation_raw_pca_axis_cosine'],
        full_head_norm_top_decile_bracket_precision=source['selectivity'][
            'full_head_norm']['validation_top_decile_bracket_precision'],
        raw_pca_axis_top_decile_bracket_precision=source['selectivity'][
            'raw_pca']['validation_top_decile_bracket_precision'])
    (ART/'joint_bracket_legibility_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PCA',summary['raw_pca'])
    print('random',random)


if __name__=='__main__':run()
