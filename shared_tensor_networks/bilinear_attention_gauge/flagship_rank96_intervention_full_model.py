"""Full-model fidelity of rank-96 QK1 cuts under QK2 branch edits."""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_full_model_eval import BondEdit,GPT,GPTConfig,get_cuts,logits,metrics
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


CUTS=("uniform_fit_l128_wtg96","corpus_fit_l128_wtg96",
      "sequence_fit_rank64_cw0")
MODES=("constant","key_shift")


def run(n_docs=24,seeds=(23,29),length=128,batch_size=8):
    torch.set_num_threads(4)
    config=json.loads((SNAPSHOT/"config.json").read_text())
    config.pop("step",None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    del state
    editor=BondEdit(model)
    available=get_cuts()
    cuts={name:torch.from_numpy(available[name]).to(torch.float32)
          for name in CUTS}
    corpora={"fineweb_test":(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];gates=[]
    with torch.no_grad():
        for corpus_name,(path,begin,end) in corpora.items():
            corpus=np.load(path,mmap_mode="r")
            for seed in seeds:
                rng=np.random.default_rng(seed+length*1000)
                docs=rng.choice(np.arange(begin,end),size=n_docs,replace=False)
                totals={mode:{name:dict(kl_sum=0.,argmax_flips=0,
                    token_count=0,max_abs_logit=0.) for name in CUTS}
                        for mode in MODES}
                shift_sums={mode:0. for mode in MODES}
                for offset in range(0,n_docs,batch_size):
                    spans=[]
                    for doc in docs[offset:offset+batch_size]:
                        start=int(rng.integers(0,corpus.shape[1]-length))
                        spans.append(np.asarray(corpus[doc,start:start+length+1],
                                                dtype=np.int64))
                    batch=np.stack(spans)
                    idx=torch.from_numpy(batch[:,:-1].copy())
                    target=torch.from_numpy(batch[:,1:].copy())
                    editor.qk2_mode="natural";editor.cut=None
                    natural=logits(model,idx)
                    for mode in MODES:
                        editor.qk2_mode=mode;editor.cut=None
                        reference=logits(model,idx)
                        shift_sums[mode]+=metrics(natural,reference,target)["kl_sum"]
                        editor.cut=torch.eye(HD)
                        identity=logits(model,idx)
                        gate=float((identity-reference).abs().max())
                        assert gate<1e-5,gate
                        gates.append(gate)
                        for name,x in cuts.items():
                            editor.cut=x
                            candidate=logits(model,idx)
                            measure=metrics(reference,candidate,target)
                            for field in ("kl_sum","argmax_flips","token_count"):
                                totals[mode][name][field]+=measure[field]
                            totals[mode][name]["max_abs_logit"]=max(
                                totals[mode][name]["max_abs_logit"],
                                measure["max_abs_logit"])
                for mode in MODES:
                    for name,stat in totals[mode].items():
                        records.append(dict(corpus=corpus_name,seed=seed,
                            intervention=mode,cut=name,n_docs=n_docs,length=length,
                            natural_to_intervened_reference_kl_per_token=
                                shift_sums[mode]/stat["token_count"],
                            intervened_reference_to_cut_kl_per_token=
                                stat["kl_sum"]/stat["token_count"],
                            flip_fraction=stat["argmax_flips"]/stat["token_count"],
                            **stat))
                print(corpus_name,seed,flush=True)
    output=dict(layer=0,head=0,length=length,n_docs=n_docs,seeds=list(seeds),
        cuts=list(CUTS),interventions=list(MODES),
        identity_cut_max_abs_logit=max(gates),
        document_windows="same document ids/starts as head0_full_model_rank96_panel_n24.json",
        intervention_definition="QK2 score in layer-0 head-0 is replaced by one or circularly shifted one key column before causal masking; other heads and layers retain checkpoint values",
        scope="full-model KL from each QK2-intervened reference to its QK1-cut counterpart; these counterfactual references can be far from natural model",
        records=records)
    (ART/"head0_rank96_intervention_full_model_n24.json").write_text(
        json.dumps(output,indent=2)+"\n")
    for corpus_name in corpora:
        for mode in MODES:
            print(corpus_name,mode,{name:float(np.mean([
                r["intervened_reference_to_cut_kl_per_token"] for r in records
                if r["corpus"]==corpus_name and r["intervention"]==mode and
                r["cut"]==name])) for name in CUTS},flush=True)
    return output


if __name__=="__main__":run()
