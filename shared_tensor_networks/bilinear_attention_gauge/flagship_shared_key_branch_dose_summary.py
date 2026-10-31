"""Paired dose summary for full-model QK2 branch edits."""

import json

import numpy as np

from flagship_transfer import ART


STEM='all_head_shared_key_branch_full_model_n16_s91-93'
FILES={0:STEM+'_a0p0.json',.25:STEM+'_a0p25.json',
       .5:STEM+'_a0p5.json',1.:STEM+'.json'}
METHODS=('usage_svd_256','natural_function_256','branch_half_256',
         'branch_only_256')
CORPORA=('fineweb_test','wikitext103','tinystories')


def run():
    panels={strength:json.loads((ART/name).read_text())
            for strength,name in FILES.items()}
    assert all(p['methods']==list(METHODS) and p['n_docs_per_seed']==16
               and p['seeds']==[91,93] and p['length']==128
               for p in panels.values())
    rng=np.random.default_rng(20261120)
    rows=[]
    for corpus in CORPORA:
        keys=None
        for strength,panel in panels.items():
            by_method={method:{(r['seed'],r['document_id'],r['start_position']):r
                        for r in panel['records'] if r['corpus']==corpus
                        and r['method']==method} for method in METHODS}
            current=sorted(by_method[METHODS[0]])
            assert len(current)==32 and all(set(by_method[m])==set(current)
                                            for m in METHODS)
            if keys is None:keys=current
            else:assert keys==current
            assert sum(k[0]==91 for k in current)==16
            assert sum(k[0]==93 for k in current)==16
            reference=by_method['natural_function_256']
            baseline=np.array([reference[k]['kl_per_token'] for k in current])
            for method in METHODS:
                part=by_method[method]
                values=np.array([part[k]['kl_per_token'] for k in current])
                diff=values-baseline
                boot=np.empty(10000)
                for j in range(len(boot)):
                    draws=np.r_[rng.integers(0,16,16),
                                16+rng.integers(0,16,16)]
                    boot[j]=diff[draws].mean()
                rows.append(dict(corpus=corpus,branch_strength=strength,
                    method=method,n_windows=32,
                    mean_kl_to_branch_reference=float(values.mean()),
                    median_kl_to_branch_reference=float(np.median(values)),
                    mean_kl_delta_vs_natural_fit=float(diff.mean()),
                    paired_wins_vs_natural_fit=int(np.sum(diff<0)),
                    bootstrap_95_mean_delta=[float(v) for v in
                                             np.quantile(boot,[.025,.975])],
                    mean_ce_delta_vs_branch_reference=float(np.mean([
                        part[k]['ce_delta'] for k in current])),
                    mean_argmax_flip_fraction=float(np.mean([
                        part[k]['argmax_flip_fraction'] for k in current])),
                    native_to_branch_reference_kl=float(np.mean([
                        part[k]['native_to_branch_kl_per_token'] for k in current])),
                    native_ce=float(np.mean([part[k]['native_ce'] for k in current])),
                    branch_reference_ce=float(np.mean([
                        part[k]['branch_reference_ce'] for k in current]))))
    zero=panels[0]
    assert all(abs(r['native_to_branch_kl_per_token'])<1e-10 and
               abs(r['native_ce']-r['branch_reference_ce'])<1e-10
               for r in zero['records'])
    assert all(p['identity_basis_max_abs_logit']<1e-3
               for p in panels.values())
    result=dict(description='Four-strength full-model branch fidelity on identical held-out 128-token windows, with paired bootstrap stratified by two sampling seeds',
        source_files=FILES,methods=METHODS,corpora=CORPORA,
        paired_window_count_per_corpus=32,
        zero_strength_exact_native_gate=True,
        identity_projection_max_abs_logit=max(p['identity_basis_max_abs_logit']
                                              for p in panels.values()),
        bootstrap_draws=10000,rows=rows)
    path=ART/'all_head_shared_key_branch_dose_summary.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    for corpus in CORPORA:
        for strength in FILES:
            r=next(x for x in rows if x['corpus']==corpus and
                   x['branch_strength']==strength and
                   x['method']=='branch_half_256')
            print(corpus,strength,'mean delta',r['mean_kl_delta_vs_natural_fit'],
                  'wins',r['paired_wins_vs_natural_fit'],
                  'CI',r['bootstrap_95_mean_delta'],flush=True)


if __name__=='__main__':run()
