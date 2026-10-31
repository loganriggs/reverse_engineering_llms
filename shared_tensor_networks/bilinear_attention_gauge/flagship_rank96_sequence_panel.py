"""Head-output and branch-edit fidelity on rank-96 full-model test windows."""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_full_model_eval import get_cuts
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_sequence_transfer import sequence_head
from flagship_transfer import ART


NAMES=("uniform_fit_l128_wtg96","corpus_fit_l128_wtg96",
       "sequence_fit_rank64_cw0")


def run(n_docs=24,seeds=(23,29),length=128,batch_size=8):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    all_cuts=get_cuts();cuts={name:all_cuts[name] for name in NAMES}
    corpora={"fineweb_test":(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    rows=[]
    for corpus_name,(path,begin,end) in corpora.items():
        corpus=np.load(path,mmap_mode="r")
        for seed in seeds:
            rng=np.random.default_rng(seed+length*1000)
            docs=rng.choice(np.arange(begin,end),size=n_docs,replace=False)
            totals={name:{mode:dict(sse=0.,energy=0.,final_sse=0.,
                                   final_energy=0.)
                          for mode in ("natural","constant","key_shift")}
                    for name in NAMES}
            for offset in range(0,n_docs,batch_size):
                for doc in docs[offset:offset+batch_size]:
                    start=int(rng.integers(0,corpus.shape[1]-length))
                    tokens=np.asarray(corpus[doc,start:start+length],dtype=np.int64)
                    for mode in ("natural","constant","key_shift"):
                        outputs=sequence_head(state,tokens,0,cuts,qk2_mode=mode)
                        reference=outputs["identity"]
                        for name in NAMES:
                            difference=outputs[name]-reference
                            stat=totals[name][mode]
                            stat["sse"]+=float(np.sum(difference**2))
                            stat["energy"]+=float(np.sum(reference**2))
                            stat["final_sse"]+=float(np.sum(difference[-1]**2))
                            stat["final_energy"]+=float(np.sum(reference[-1]**2))
            for name in NAMES:
                for mode,stat in totals[name].items():
                    rows.append(dict(corpus=corpus_name,seed=seed,cut=name,
                        mode=mode,n_docs=n_docs,
                        sequence_head_relative_mse=stat["sse"]/stat["energy"],
                        final_head_relative_mse=stat["final_sse"]/stat["final_energy"],
                        **stat))
            print(corpus_name,seed,flush=True)
    result=dict(head=0,layer=0,length=length,n_docs=n_docs,seeds=list(seeds),
        cuts=list(NAMES),
        document_windows="exact same document ids and starts as head0_full_model_rank96_panel_n24.json",
        scope="summed layer-0 head-output fidelity under natural, QK2-constant, and QK2-key-shift conditions; no downstream propagation for branch edits",
        rows=rows)
    (ART/"head0_rank96_sequence_panel_n24.json").write_text(json.dumps(result,indent=2)+"\n")
    for corpus_name in corpora:
        for mode in ("natural","constant","key_shift"):
            print(corpus_name,mode,{name:float(np.mean([
                r["sequence_head_relative_mse"] for r in rows
                if r["corpus"]==corpus_name and r["mode"]==mode and r["cut"]==name]))
                for name in NAMES},flush=True)
    return result


if __name__=="__main__":run()
