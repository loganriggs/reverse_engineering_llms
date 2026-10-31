"""Paired document bootstrap for rare-key local and full-model panels."""

import json

import numpy as np

from flagship_transfer import ART


PRIORS=('uniform','mixed','corpus')
CORPORA=('fineweb_test','wikitext103','tinystories')


def interval(values):
    return [float(x) for x in np.quantile(values,[.025,.975])]


def analyze_local(seed=20264150,n_boot=10000):
    rng=np.random.default_rng(20264200)
    panels={fit:json.loads((ART/f'head0_pre_rank96_rare_key_local_fit{fit}_n128_s{seed}.json').read_text())
            for fit in (41,43,47)}
    assert all(panels[41]['cases']==panels[fit]['cases'] for fit in (43,47))
    summaries={}
    for fit,panel in panels.items():
        lookup={(r['case_index'],r['prior']):r for r in panel['records']}
        summaries[str(fit)]={}
        for corpus in CORPORA:
            summaries[str(fit)][corpus]={}
            for stratum in ('rare','common'):
                indices=[i for i,c in enumerate(panel['cases'])
                         if c['corpus']==corpus and c['stratum']==stratum]
                assert len(indices)==128
                energy=np.array([lookup[i,'uniform']['native_key_output_energy']
                                 for i in indices])
                error={prior:np.array([lookup[i,prior]['removed_post_c_proj_energy']
                                      for i in indices]) for prior in PRIORS}
                draws=rng.integers(0,len(indices),size=(n_boot,len(indices)))
                denominator=energy[draws].sum(axis=1)
                methods={prior:dict(relative_output_error=float(values.sum()/energy.sum()),
                    bootstrap_interval=interval(values[draws].sum(axis=1)/denominator),
                    median_per_case_relative_error=float(np.median(values/
                        np.maximum(energy,1e-30))))
                    for prior,values in error.items()}
                paired={}
                for a,b in (('mixed','uniform'),('mixed','corpus'),('corpus','uniform')):
                    diff=error[a]-error[b]
                    boot=diff[draws].sum(axis=1)/denominator
                    paired[f'{a}_minus_{b}']=dict(
                        relative_error_difference=float(diff.sum()/energy.sum()),
                        bootstrap_interval=interval(boot),
                        wins=int(np.sum(error[a]<error[b])),
                        n=len(indices))
                summaries[str(fit)][corpus][stratum]=dict(
                    n=len(indices),energy_effective_n=float(energy.sum()**2/
                                                             np.sum(energy*energy)),
                    methods=methods,paired=paired)
    return summaries


def analyze_full(path,n_expected,strata=('rare','common'),n_boot=10000):
    rng=np.random.default_rng(20264201)
    panel=json.loads((ART/path).read_text())
    lookup={(r['case_index'],r['prior']):r for r in panel['records']}
    summaries={}
    for corpus in CORPORA:
        summaries[corpus]={}
        for stratum in strata:
            indices=[i for i,c in enumerate(panel['cases'])
                     if c['corpus']==corpus and c['stratum']==stratum]
            assert len(indices)==n_expected
            values={prior:np.array([lookup[i,prior]['suffix_kl'] for i in indices])
                    for prior in PRIORS}
            draws=rng.integers(0,len(indices),size=(n_boot,len(indices)))
            methods={prior:dict(mean_suffix_kl=float(x.mean()),
                median_suffix_kl=float(np.median(x)),
                bootstrap_mean_interval=interval(x[draws].mean(axis=1)),
                kl_effective_n=float(x.sum()**2/np.sum(x*x)))
                for prior,x in values.items()}
            paired={}
            for a,b in (('mixed','uniform'),('mixed','corpus'),('corpus','uniform')):
                diff=values[a]-values[b]
                paired[f'{a}_minus_{b}']=dict(
                    mean_difference=float(diff.mean()),
                    bootstrap_mean_interval=interval(diff[draws].mean(axis=1)),
                    wins=int(np.sum(values[a]<values[b])),n=len(indices))
            summaries[corpus][stratum]=dict(n=len(indices),methods=methods,paired=paired)
    return summaries


def run():
    local=analyze_local()
    full=analyze_full('head0_pre_rank96_rare_key_transfer_n24_s20264100.json',24)
    expanded=analyze_full('head0_pre_rank96_rare_key_transfer_n64_s20264300_rare.json',
                          64,('rare',))
    result=dict(description='Paired document bootstrap: 10,000 resamples per corpus/stratum, 128 natural selected-key cases for each of three fixed rank-96 cut fit seeds in exact immediate output space, 24 cases per common/rare stratum for fit-seed-41 full 18-layer suffix KL, and a fresh 64-case rare-only full-model panel. Same cases across cut methods, and same local cases across fit seeds. Bootstrap over documents; fit seeds are not independent documents.',
                local=local,full_model_24=full,full_model_rare_64=expanded)
    (ART/'head0_pre_rank96_rare_key_transfer_analysis.json').write_text(
        json.dumps(result,indent=2)+'\n')
    for corpus in CORPORA:
        for stratum in ('rare','common'):
            print(corpus,stratum,'local seed41',
                {k:round(v['relative_output_error'],4)
                 for k,v in local['41'][corpus][stratum]['methods'].items()},
                'full KL',
                {k:format(v['mean_suffix_kl'],'.2e')
                 for k,v in full[corpus][stratum]['methods'].items()})
    return result


if __name__=='__main__':run()
