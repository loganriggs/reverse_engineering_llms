"""Bootstrap matched natural-pair comparisons for effective-score bases."""

from __future__ import annotations

import json

import numpy as np

from flagship_transfer import ART


def run(replicates=4000):
    score=json.loads((ART/"head0_effective_score_natural_pairs.json").read_text())
    flat=json.loads((ART/"head0_full_function_natural_pairs.json").read_text())
    records=score["records"]+[r for r in flat["records"]
                              if r["method"]=="bond_rank64"]
    methods=sorted(set(r["method"] for r in records))
    rng=np.random.default_rng(7811)
    rows=[]
    for corpus in ("fineweb_test","wikitext103","tinystories"):
        reference=[r for r in records if r["corpus"]==corpus and
                   r["method"]=="bond_rank64"]
        n=len(reference)
        energy=np.array([r["reference_energy"] for r in reference])
        ids=np.array([(r["document_id"],r["start"],r["query_id"],r["key_id"])
                      for r in reference])
        draw=rng.integers(0,n,size=(replicates,n))
        for method in methods:
            selected=[r for r in records if r["corpus"]==corpus and
                      r["method"]==method]
            assert len(selected)==n
            selected_ids=np.array([(r["document_id"],r["start"],
                                    r["query_id"],r["key_id"])
                                   for r in selected])
            assert np.array_equal(ids,selected_ids)
            error=np.array([r["squared_error"] for r in selected])
            bootstrap=np.sum(error[draw],axis=1)/np.sum(energy[draw],axis=1)
            rows.append(dict(corpus=corpus,method=method,n=n,
                relative_mse=float(error.sum()/energy.sum()),
                paired_bootstrap_p025=float(np.quantile(bootstrap,.025)),
                paired_bootstrap_p975=float(np.quantile(bootstrap,.975)),
                reference_energy_effective_sample_size=float(energy.sum()**2/
                                                              np.sum(energy**2))))
    result=dict(replicates=replicates,seed=7811,
        scope="paired bootstrap of same documents and token pairs for effective-score and structured bond cuts; no full-model behavior",
        rows=rows)
    (ART/"head0_effective_score_pair_bootstrap.json").write_text(json.dumps(result,indent=2)+"\n")
    for corpus in ("fineweb_test","wikitext103","tinystories"):
        print(corpus,flush=True)
        for name in ("strict_uniform_rank64","strict_mostly_corpus_rank64",
                     "mixed_mostly_corpus_rank64","multi_mostly_corpus_rank64",
                     "bond_rank64"):
            row=next(r for r in rows if r["corpus"]==corpus and r["method"]==name)
            print(name,row["relative_mse"],row["paired_bootstrap_p025"],
                  row["paired_bootstrap_p975"],flush=True)
    return result


if __name__=="__main__":run()
