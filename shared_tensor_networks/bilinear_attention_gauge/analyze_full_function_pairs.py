"""Energy concentration and paired bootstrap for natural pair-response errors."""

from __future__ import annotations

import json

import numpy as np

from flagship_transfer import ART


def run(replicates=4000):
    source=json.loads((ART/"head0_full_function_natural_pairs.json").read_text())
    records=source["records"]
    rng=np.random.default_rng(6149)
    output=[]
    for corpus in ("fineweb_test","wikitext103","tinystories"):
        by_method={method:[r for r in records if r["corpus"]==corpus and
                                     r["method"]==method]
                   for method in ("uniform_rank64","half_corpus_rank64",
                                  "mostly_corpus_rank64","bond_rank64")}
        n=len(by_method["bond_rank64"])
        energy=np.array([r["reference_energy"] for r in by_method["bond_rank64"]])
        rng_index=rng.integers(0,n,size=(replicates,n))
        for method,rows in by_method.items():
            error=np.array([r["squared_error"] for r in rows])
            assert len(error)==n
            bootstrap=np.sum(error[rng_index],axis=1)/np.sum(energy[rng_index],axis=1)
            top=int(np.argmax(energy))
            leave=np.arange(n)!=top
            output.append(dict(corpus=corpus,method=method,n=n,
                relative_mse=float(error.sum()/energy.sum()),
                paired_bootstrap_p025=float(np.quantile(bootstrap,.025)),
                paired_bootstrap_p975=float(np.quantile(bootstrap,.975)),
                reference_energy_effective_sample_size=float(energy.sum()**2/
                                                              np.sum(energy**2)),
                largest_pair_energy_fraction=float(energy.max()/energy.sum()),
                relative_mse_without_largest_energy_pair=float(error[leave].sum()/
                                                                 energy[leave].sum())))
    result=dict(replicates=replicates,seed=6149,
        scope="paired document bootstrap of fixed-position actual adjacent pair-response errors; energy concentration can make intervals wide",
        rows=output)
    (ART/"head0_full_function_pair_bootstrap.json").write_text(json.dumps(result,indent=2)+"\n")
    for row in output:print(row,flush=True)
    return result


if __name__=="__main__":run()
