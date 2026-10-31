"""Exploratory full-model test of a post-hoc ' as' -> ' much' motif."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_ablation import selected_cuts
from flagship_full_model_eval import BondEdit,GPT,GPTConfig,logits
from flagship_key_motif import GAUGES,windows
from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


def pointwise_kl(reference,candidate):
    p=F.log_softmax(reference.to(torch.float64),dim=-1)
    q=F.log_softmax(candidate.to(torch.float64),dim=-1)
    return (p.exp()*(p-q)).sum(-1).numpy()


def run():
    torch.set_num_threads(1)
    config=json.loads((SNAPSHOT/"config.json").read_text())
    config.pop("step",None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    editor=BondEdit(model)
    base,full_cuts,_=selected_cuts(2,"key")
    prior=json.loads((ART/"head0_component_ablation_n8_matched_key_panel.json").read_text())
    cuts={name:base+prior["details"][name]["component_deletion_fraction"]*
          (full_cuts[name]-base) for name in GAUGES}
    base=torch.from_numpy(base).to(torch.float32)
    cuts={name:torch.from_numpy(value).to(torch.float32)
          for name,value in cuts.items()}
    corpora={"fineweb_test":(CORPUS,17500,20000,16),
             **{name:(path,0,1200,32) for name,path in DATASETS.items()}}
    records=[];counts={}
    with torch.no_grad():
        for corpus_name,(path,begin,end,maximum) in corpora.items():
            corpus=np.load(path,mmap_mode="r")
            cases,counts[corpus_name]=windows(corpus,begin,end,91000+begin,
                                              max_motif=maximum)
            for offset in range(0,len(cases),4):
                batch=cases[offset:offset+4]
                tokens=torch.from_numpy(np.stack([c["tokens"] for c in batch]))
                editor.cut=base
                reference=logits(model,tokens)
                for name,cut in cuts.items():
                    editor.cut=cut
                    changed=logits(model,tokens)
                    kl=pointwise_kl(reference,changed)
                    for index,case in enumerate(batch):
                        query=case["query_position"]
                        records.append(dict(corpus=corpus_name,
                            document_id=case["doc"],kind=case["kind"],
                            motif_document_position=case["position"],
                            window_start=case["start"],
                            query_position=query,gauge=name,
                            mean_kl=float(np.mean(kl[index])),
                            query_kl=float(kl[index,query]),
                            max_kl=float(np.max(kl[index]))))
            print(corpus_name,counts[corpus_name],flush=True)
    output=dict(motif="post-hoc ' as' -> ' much' from one FineWeb test outlier",
        cuts="same pre-calibrated partial deletions; no refit or reselection",
        fineweb_test_document_range=[17500,20000],
        external_document_range=[0,1200],
        scope="exploratory targeted-context test, not an unbiased population estimate or semantic discovery",
        counts=counts,records=records)
    (ART/"head0_key_sparse_motif_full_model.json").write_text(json.dumps(output,indent=2)+"\n")
    for corpus_name in corpora:
        for gauge in GAUGES:
            selected=[r for r in records if r["corpus"]==corpus_name and
                      r["gauge"]==gauge]
            if not selected:continue
            mot=np.array([r["query_kl"] for r in selected if r["kind"]=="motif"])
            control=np.array([r["query_kl"] for r in selected if r["kind"]=="control"])
            print(corpus_name,gauge,"query KL motif",float(np.median(mot)),
                  "control",float(np.median(control)),flush=True)
    return output


if __name__=="__main__":run()
