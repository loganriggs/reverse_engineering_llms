#!/usr/bin/env python3
"""Selection and permutation red-team for dynamic bilinear-neuron roles."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from run_bilin18_layer01_dynamic_neuron_roles import pair_aggregate
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer01_dynamic_neuron_role_redteam.json")
OUTPUT=TENSOR_ROOT/"layer01_dynamic_neuron_role_redteam.json"
REPORT=TENSOR_ROOT/"LAYER01_DYNAMIC_NEURON_ROLE_REDTEAM.md"


def bootstrap(values,samples,rng):
    values=np.asarray(values); draws=np.median(values[rng.integers(0,len(values),size=(samples,len(values)))],axis=1)
    return [float(value) for value in np.quantile(draws,[.025,.5,.975])]


def recurring_side_test(records,minimum,samples,rng):
    by_channel=collections.defaultdict(list)
    for row in records:
        for index,side in zip(row["indices"],row["sides"]): by_channel[index].append(bool(side))
    recurring={channel:values for channel,values in by_channel.items() if len(values)>=minimum}
    def statistic(mapping): return float(np.median([max(np.mean(values),1-np.mean(values)) for values in mapping.values()]))
    observed=statistic(recurring); channels=list(recurring); lengths=[len(recurring[channel]) for channel in channels]; pool=np.asarray([value for channel in channels for value in recurring[channel]],dtype=bool); null=[]
    for _ in range(samples):
        shuffled=rng.permutation(pool); offset=0; mapping={}
        for channel,length in zip(channels,lengths): mapping[channel]=shuffled[offset:offset+length]; offset+=length
        null.append(statistic(mapping))
    return observed,(1+sum(value>=observed for value in null))/(samples+1),[float(value) for value in np.quantile(null,[.01,.5,.99])],len(recurring)


def cross_corpus_test(records,rank,key,samples,rng):
    pair_values=[]
    for pair in sorted({row["pair"] for row in records}):
        left=pair_aggregate([row for row in records if row["corpus"]=="fineweb" and row["pair"]==pair],rank); right=pair_aggregate([row for row in records if row["corpus"]=="tinystories" and row["pair"]==pair],rank); common=sorted(set(left)&set(right)); pair_values.append((np.asarray([left[index][key] for index in common],dtype=bool),np.asarray([right[index][key] for index in common],dtype=bool)))
    statistic=lambda rows:float(np.median([np.mean(left==right) for left,right in rows if len(left)]))
    observed=statistic(pair_values); null=[]
    for _ in range(samples): null.append(statistic([(left,rng.permutation(right)) for left,right in pair_values]))
    return observed,(1+sum(value>=observed for value in null))/(samples+1),[float(value) for value in np.quantile(null,[.01,.5,.99])]


def main():
    config=json.loads(CONFIG.read_text()); source=json.loads(Path(config["source_output"]).read_text()); rng=np.random.default_rng(config["seed"]); results={}
    for layer,layer_row in source["layers"].items():
        results[layer]={}
        for corpus in ("fineweb","tinystories"):
            rows=[row for row in layer_row["records"] if row["corpus"]==corpus]; by_pair=collections.defaultdict(list)
            for row in rows: by_pair[row["pair"]].append(row["selection_baselines"]["contribution"]["single_sided"]-row["selection_baselines"]["all"]["single_sided"])
            pair_differences=[float(np.median(values)) for values in by_pair.values()]; observed,pvalue,null,count=recurring_side_test(rows,config["recurrence_minimum"],config["permutation_samples"],rng); results[layer][corpus]={"asymmetry_enrichment":float(np.median(pair_differences)),"asymmetry_bootstrap_ci":bootstrap(pair_differences,config["bootstrap_samples"],rng),"recurring_side_agreement":observed,"recurring_shuffle_p":pvalue,"recurring_null_q01_median_q99":null,"recurring_channels":count}
        results[layer]["cross_corpus_side"]={}; observed,pvalue,null=cross_corpus_test(layer_row["records"],source["config"]["rank"],"side",config["permutation_samples"],rng); results[layer]["cross_corpus_side"]={"observed":observed,"p":pvalue,"null_q01_median_q99":null}
        observed,pvalue,null=cross_corpus_test(layer_row["records"],source["config"]["rank"],"sign",config["permutation_samples"],rng); results[layer]["cross_corpus_sign"]={"observed":observed,"p":pvalue,"null_q01_median_q99":null}
    cells=[results[str(layer)][corpus] for layer in source["config"]["layers"] for corpus in ("fineweb","tinystories")]
    hypotheses={"N0_asymmetry_is_enriched_by_causal_selection":{"supported":all(row["asymmetry_enrichment"]>=.03 and row["asymmetry_bootstrap_ci"][0]>0 for row in cells)},"N1_recurring_side_roles_beat_shuffle":{"supported":all(row["recurring_shuffle_p"]<.05 for row in cells)},"N2_cross_corpus_side_roles_beat_shuffle":{"supported":all(results[str(layer)]["cross_corpus_side"]["p"]<.05 for layer in source["config"]["layers"])},"N3_cross_corpus_sign_stability_beats_shuffle":{"supported":all(results[str(layer)]["cross_corpus_sign"]["p"]<.05 for layer in source["config"]["layers"])}}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2)); lines=["# Red-team of dynamic bilinear-neuron factor roles","","Pair-bootstrap enrichment controls for the possibility that bilinear products are generically one-sided. Side labels are shuffled across recurring channels while preserving occurrence counts. Cross-corpus side and sign labels are shuffled within each lexical-pair intersection, preserving marginal imbalance.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Asymmetry enrichment (95% pair CI) | Recurring side agreement / null median / p |","|---:|---|---:|---:|"]
    for layer in source["config"]["layers"]:
        for corpus in ("fineweb","tinystories"):
            row=results[str(layer)][corpus]; ci=row["asymmetry_bootstrap_ci"]; lines.append(f"| {layer} | {corpus} | {row['asymmetry_enrichment']:.3f} ({ci[0]:.3f}, {ci[2]:.3f}) | {row['recurring_side_agreement']:.3f} / {row['recurring_null_q01_median_q99'][1]:.3f} / {row['recurring_shuffle_p']:.4f} |")
    lines += ["","| Layer | Cross-corpus side observed/null/p | Cross-corpus sign observed/null/p |","|---:|---:|---:|"]
    for layer in source["config"]["layers"]:
        side=results[str(layer)]["cross_corpus_side"]; sign=results[str(layer)]["cross_corpus_sign"]; lines.append(f"| {layer} | {side['observed']:.3f}/{side['null_q01_median_q99'][1]:.3f}/{side['p']:.4f} | {sign['observed']:.3f}/{sign['null_q01_median_q99'][1]:.3f}/{sign['p']:.4f} |")
    lines += ["","The sign test is essential because 97% of contribution-selected coordinates are positive; raw agreement near one can therefore be a class-imbalance artifact. Factor-side names are arbitrary under a per-neuron swap, but agreement and its shuffle null are invariant to a fixed swap."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
