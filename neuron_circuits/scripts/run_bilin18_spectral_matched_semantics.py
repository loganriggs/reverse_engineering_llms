#!/usr/bin/env python3
"""Activation-matched semantic-route test for known spectral feature paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from redteam_bilin18_spectral_local_inputs import evaluate
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_atlas import ARRAYS as ATLAS_ARRAYS, OUTPUT as ATLAS_OUTPUT
from run_bilin18_spectral_spike_pilot import ARRAYS as PILOT_ARRAYS, SPARSE, TENSOR_ROOT, load_feature_directions
from train_bilin18_causal_cp import prefix_to_layer1


CONFIG = Path("configs/bilin18_spectral_matched_semantics.json")
OUTPUT = TENSOR_ROOT / "spectral_matched_semantics.json"
REPORT = TENSOR_ROOT / "SPECTRAL_MATCHED_SEMANTICS.md"
FIGURE = TENSOR_ROOT / "spectral_matched_semantics.png"


def stable_rng(seed, text):
    return np.random.default_rng(seed + int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))


def greedy_match(targets, controls):
    remaining = set(range(len(controls))); matched = []
    for target in targets:
        index = min(remaining, key=lambda i: abs(controls[i]["feature_preactivation"] - target["feature_preactivation"]))
        remaining.remove(index); matched.append(controls[index])
    return matched


@torch.inference_mode()
def collect(model, rows, categories, features, encoder_weight, encoder_bias,
            mean, std, config, tokenizer):
    flags = np.stack([token_flags(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    category_index = {name: i for i, name in enumerate(CATEGORIES)}
    pools = {category: {True: [], False: []} for category in categories}
    positions = config["positions"]; positions_tensor = torch.as_tensor(positions, device="cuda")
    for start in range(0, len(rows), 8):
        raw = np.asarray(rows[start:start + 8, :128], dtype=np.int64); ids = torch.as_tensor(raw, device="cuda")
        _, _, _, normalized = prefix_to_layer1(model, ids)
        product = model.blocks[1].mlp.Left(normalized[:, positions_tensor]) * model.blocks[1].mlp.Right(normalized[:, positions_tensor])
        pre = F.linear((product.float().flatten(0, 1) - mean) / std, encoder_weight, encoder_bias)
        top = torch.topk(F.relu(pre), 128, dim=1).indices
        token_ids = raw[:, positions].reshape(-1)
        for category in categories:
            feature = features[category]; active = (top == feature).any(1) & (pre[:, feature] > 0)
            surface = flags[token_ids, category_index[category]].astype(bool)
            for index in torch.nonzero(active).flatten().tolist():
                batch_index, position_index = divmod(index, len(positions))
                pools[category][bool(surface[index])].append({
                    "row": start + batch_index, "position": int(positions[position_index]),
                    "score": float(pre[index, feature]), "feature_preactivation": float(pre[index, feature]),
                    "token_id": int(token_ids[index]), "surface_label": bool(surface[index]),
                    "feature_active": True,
                })
    size = config["pairs_per_stratum"]; selected = {}; excluded = {}; match_quality = {}
    for category, values in pools.items():
        positives = sorted(values[True], key=lambda row: row["score"], reverse=True)
        negatives = sorted(values[False], key=lambda row: row["score"], reverse=True)
        if len(positives) < 2 * size or len(negatives) < 2 * size:
            excluded[category] = {"active_surface_positive": len(positives), "active_surface_negative": len(negatives)}; continue
        high_positive = positives[:size]
        rng = stable_rng(config["seed"], category)
        random_positive = [positives[size:][i] for i in rng.choice(len(positives)-size, size=size, replace=False)]
        high_negative = greedy_match(high_positive, negatives)
        used = {(row["row"], row["position"]) for row in high_negative}
        remaining_negative = [row for row in negatives if (row["row"], row["position"]) not in used]
        random_negative = greedy_match(random_positive, remaining_negative)
        selected[category] = []
        for pair_group, positive_rows, negative_rows in (
                ("high", high_positive, high_negative), ("random", random_positive, random_negative)):
            for pair_id, (positive, negative) in enumerate(zip(positive_rows, negative_rows)):
                selected[category].append({**positive, "stratum": f"{pair_group}_positive", "pair_id": pair_id})
                selected[category].append({**negative, "stratum": f"{pair_group}_negative", "pair_id": pair_id})
        match_quality[category] = {}
        for group, positive_rows, negative_rows in (("high", high_positive, high_negative), ("random", random_positive, random_negative)):
            p = np.asarray([row["feature_preactivation"] for row in positive_rows]); n = np.asarray([row["feature_preactivation"] for row in negative_rows])
            pooled = np.sqrt((p.var(ddof=1) + n.var(ddof=1)) / 2).clip(1e-12)
            match_quality[category][group] = {"standardized_mean_difference": float(abs(p.mean()-n.mean())/pooled),
                                               "median_absolute_pair_difference": float(np.median(abs(p-n)))}
    counts = {category: {"positive": len(rows[True]), "negative": len(rows[False])} for category, rows in pools.items()}
    return selected, excluded, counts, match_quality


def main():
    config = json.loads(CONFIG.read_text()); atlas = json.loads(ATLAS_OUTPUT.read_text())
    pilot_arrays = np.load(PILOT_ARRAYS); atlas_arrays = np.load(ATLAS_ARRAYS)
    square = torch.from_numpy(pilot_arrays["covariance_square"]).cuda(); inverse_square = torch.from_numpy(pilot_arrays["covariance_inverse_square"]).cuda()
    categories = config["categories"]; all_categories = atlas["config"]["reference_categories"] + atlas["config"]["prospective_categories"]
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True); down = state["transformer.h.1.mlp.Down.weight"].float().cuda(); del state
    archive = np.load(SPARSE / "natural_large.npz"); train = torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32)); mean=train.mean(0).cuda(); std=train.std(0).clamp_min(1e-4).cuda(); del train,archive
    directions, metadata = load_feature_directions({"strong_categories": all_categories, "sae_seeds": atlas["config"]["feature_seeds"], "seed": atlas["config"]["seed"]}, std.cpu(), down)
    sae=torch.load(SPARSE/"activation_sae_large_seed101.pt",map_location="cpu",weights_only=True); ew=sae["encoder.weight"].float().cuda(); eb=sae["encoder.bias"].float().cuda(); decoder=sae["decoder.weight"].float()
    features={};decoders={}
    for category in categories:
        name=f"semantic/seed101/{category}";features[category]=metadata[name]["feature"];decoders[name]=std.cpu()*decoder[:,features[category]]
    del sae,decoder
    model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval();tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    all_rows=np.load(DEFAULT_CORPUS,mmap_mode="r");panel=config["panel"];rows=all_rows[panel["start"]:panel["start"]+panel["rows"]]
    sites,excluded,counts,match_quality=collect(model,rows,categories,features,ew,eb,mean,std,config,tokenizer)
    eligible=list(sites);print("matched semantic categories",eligible,"excluded",excluded,flush=True)
    local_config={"rank":config["rank"],"bases":config["bases"],"input_norm_fractions":[config["input_norm_fraction"]]}
    results=evaluate(model,rows,sites,eligible,set(categories),directions,decoders,pilot_arrays,atlas_arrays,square,inverse_square,local_config)
    fraction=str(config["input_norm_fraction"]);summaries={};paired_gaps={}
    for basis in config["bases"]:
        summaries[basis]={};paired_gaps[basis]={}
        for group in ("high","random"):
            pos_top=[];neg_top=[];pos_low=[];category_gaps=[]
            for row in results.values():
                pos=[s for s in row["sites"] if s["stratum"]==f"{group}_positive"];neg=[s for s in row["sites"] if s["stratum"]==f"{group}_negative"]
                ptop=float(np.median([s["bases"][basis][fraction]["top"]["sae_decoder_fidelity"] for s in pos])); ntop=float(np.median([s["bases"][basis][fraction]["top"]["sae_decoder_fidelity"] for s in neg])); plow=float(np.median([s["bases"][basis][fraction]["low"]["sae_decoder_fidelity"] for s in pos]))
                pos_top.append(ptop);neg_top.append(ntop);pos_low.append(plow);category_gaps.append(ptop-ntop)
            summaries[basis][group]={"positive_top":float(np.median(pos_top)),"negative_top":float(np.median(neg_top)),"positive_low":float(np.median(pos_low)),"semantic_gap":float(np.median(pos_top)-np.median(neg_top)),"top_low_gap":float(np.median(pos_top)-np.median(pos_low)),"category_gaps":category_gaps}
            paired_gaps[basis][group]=category_gaps
    match_max={group:max(row[group]["standardized_mean_difference"] for row in match_quality.values()) for group in ("high","random")}
    hypotheses={
      "M1_matched_semantic_route":{"supported":any(all(summaries[b][g]["semantic_gap"]>=.15 for g in ("high","random")) for b in config["bases"]),"summaries":summaries},
      "M2_positive_path_specificity":{"supported":all(summaries[b][g]["top_low_gap"]>=.20 for b in config["bases"] for g in ("high","random")),"summaries":summaries},
      "M3_activation_match_quality":{"supported":all(v<=.25 for v in match_max.values()),"maximum_standardized_differences":match_max},
      "M4_cross_basis_semantic_gap":{"supported":any(all(v>0 for v in paired_gaps[b][g]) for b in config["bases"] for g in ("high","random")),"category_gaps":paired_gaps},
      "M5_random_positive_generality":{"supported":any(summaries[b]["random"]["positive_top"]>=.25 for b in config["bases"]),"by_basis":{b:summaries[b]["random"]["positive_top"] for b in config["bases"]}}
    }
    output={"config":config,"excluded":excluded,"active_counts":counts,"match_quality":match_quality,"sites":sites,"results":results,"summaries":summaries,"hypotheses":hypotheses};OUTPUT.write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=["# Activation-matched semantic-route test","","Active surface-positive sites are paired to active surface-negative sites with nearest SAE feature preactivation. This tests whether the category-specific spectral modes identify a distinct route into a shared active feature, rather than merely tracking activation magnitude.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Matched semantics](spectral_matched_semantics.png)","","| Basis | Stratum | Positive top | Matched-negative top | Positive low | Semantic gap |","|---|---|---:|---:|---:|---:|"]
    for basis in config["bases"]:
      for group in ("high","random"):
       row=summaries[basis][group];lines.append(f"| {basis} | {group} | {row['positive_top']:.3f} | {row['negative_top']:.3f} | {row['positive_low']:.3f} | {row['semantic_gap']:.3f} |")
    lines += ["", "All interventions are rank-4, float32, and 10% of input norm. Per-category matching diagnostics and site-level effects are in `spectral_matched_semantics.json`."];REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(11,4.3));x=np.arange(2);groups=["high","random"]
    for axis,basis in zip(axes,config["bases"]):
      axis.bar(x-.2,[summaries[basis][g]["positive_top"] for g in groups],.2,label="surface positive/top")
      axis.bar(x,[summaries[basis][g]["negative_top"] for g in groups],.2,label="matched negative/top")
      axis.bar(x+.2,[summaries[basis][g]["positive_low"] for g in groups],.2,label="positive/low control")
      axis.set_xticks(x,groups);axis.set_ylabel("SAE-decoder mediation fidelity");axis.set_title(basis);axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig);print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
