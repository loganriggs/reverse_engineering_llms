"""Paired context bootstrap for frozen rank-four natural-text KL panel."""

import json

import numpy as np

from flagship_transfer import ART


SEED = 20261117
BOOTSTRAPS = 10000


def run():
    source = json.loads((ART / 'later_head_rank4_kl_scale.json').read_text())
    rng = np.random.default_rng(SEED)
    panels = []
    aggregate = {}
    for target in source['targets']:
        deltas = []
        for length in (32, 64):
            for corpus in ('fineweb_test', 'wikitext103', 'tinystories'):
                rows = {r['method']: r for r in source['records']
                        if r['target'] == target and r['length'] == length and r['corpus'] == corpus}
                assert set(rows) == set(source['measures'])
                n = rows['weight']['count']
                assert all(len(row['kl']) == n for row in rows.values())
                difference = np.asarray(rows['corpus']['kl']) - np.asarray(rows['weight']['kl'])
                indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
                boot = difference[indices].mean(axis=1)
                deltas.append(boot)
                panels.append(dict(target=target, corpus=corpus, length=length, count=n,
                    mean_kl={name:row['mean_kl'] for name,row in rows.items()},
                    top1_flips={name:row['flip_count'] for name,row in rows.items()},
                    corpus_minus_weight_mean_kl=float(difference.mean()),
                    corpus_minus_weight_mean_kl_ci95=np.quantile(boot,[.025,.975]).tolist(),
                    fraction_contexts_corpus_lower_kl=float(np.mean(difference<0))))
        joint = np.mean(np.stack(deltas), axis=0)
        aggregate[target] = dict(equal_panel_mean_corpus_minus_weight_kl=float(joint.mean()),
            equal_panel_mean_corpus_minus_weight_kl_ci95=np.quantile(joint,[.025,.975]).tolist(),
            caveat='The two context lengths may sample some of the same source documents; panel bootstrap does not account for cross-panel document overlap.')
    result = dict(description='Paired within-panel context bootstrap of corpus versus weight frozen rank-four KL; positive difference favors weight',
        source='later_head_rank4_kl_scale.json', seed=SEED, bootstrap_replicates=BOOTSTRAPS,
        panels=panels, aggregate=aggregate)
    (ART/'later_head_rank4_kl_scale_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in panels:
        print(row['target'],row['length'],row['corpus'],row['count'],
              round(row['mean_kl']['weight'],6),round(row['mean_kl']['corpus'],6),
              np.round(row['corpus_minus_weight_mean_kl_ci95'],6),flush=True)
    print('Aggregate',aggregate,flush=True)


if __name__=='__main__':
    run()
