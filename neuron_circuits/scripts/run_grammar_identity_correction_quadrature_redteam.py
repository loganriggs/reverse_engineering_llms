#!/usr/bin/env python3
"""Thirty-two-point convergence red team for identity-correction tensor paths."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from run_grammar_identity_correction_tensor_paths import analyze


CONFIG = Path("configs/grammar_identity_correction_quadrature_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_identity_correction_quadrature_redteam.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_IDENTITY_CORRECTION_QUADRATURE_REDTEAM.md")


def main():
    config = json.loads(CONFIG.read_text()); parent = json.loads(Path(config["parent_output"]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"; runs = []
    for spec in config["models"]:
        print("q32", spec["seed"], flush=True); runs.append(analyze(spec, config, device))
    first, held = map(str, config["evaluation_templates"])
    def val(run, status, template, corruption, method, key="median_fidelity"):
        return run["results"][status][template][corruption][method][key]
    def old(seed): return next(run for run in parent["runs"] if run["seed"] == seed)
    r0 = all(max(run["maximum_errors"].values()) < 1e-4 and val(run,status,template,corruption,"top_729") >= .9999 for run in runs for status in ("unselected","selected") for template in (first,held) for corruption in ("mean","roll"))
    r1 = all(abs(val(run,status,template,corruption,f"top_{count}") - val(old(run["seed"]),status,template,corruption,f"top_{count}")) <= .03 for run in runs for status in ("unselected","selected") for template in (first,held) for corruption in ("mean","roll") for count in (32,128,256))
    r2 = all(abs(val(run,status,template,corruption,"contains_bilinear","signed_projection") - val(old(run["seed"]),status,template,corruption,"contains_bilinear","signed_projection")) <= .03 for run in runs for status in ("unselected","selected") for template in (first,held) for corruption in ("mean","roll"))
    r3 = all(val(run,"selected",held,corruption,"top_128") >= .95 for run in runs for corruption in ("mean","roll"))
    r4 = all(val(run,"selected",held,"roll","top_256") >= .75 for run in runs)
    r5 = all(val(run,"selected",held,"mean","top_128") >= run["results"]["selected"][held]["mean"]["random_128"]["median_fidelity_p95"] + .10 for run in runs)
    r6 = all(abs(val(run,"selected",first,"mean","top_256") - val(run,"selected",held,"mean","top_256")) <= .10 for run in runs)
    r7 = all(row["effectful_examples"] >= 64 for run in runs for status in ("unselected","selected") for template in (first,held) for corruption in ("mean","roll") for row in run["results"][status][template][corruption].values() if isinstance(row,dict) and "effectful_examples" in row)
    hypotheses = {name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(r0,r1,r2,r3,r4,r5,r6,r7))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Thirty-two-point convergence red team for identity tensor paths","",
           "The retained eight-point run failed its frozen exactness gate on rare large wrong-codeword interventions. This rerun changes only Gauss--Legendre order from 8 to 32. It compares the resulting finite path metrics against the original artifact rather than replacing it.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Seed | max state/logit error | selected mean top32/128/256 | selected roll top32/128/256 | bilinear projection mean/roll |","|---:|---:|---:|---:|---:|"]
    for run in runs:
        mean=run["results"]["selected"][held]["mean"];roll=run["results"]["selected"][held]["roll"];err=run["maximum_errors"]
        triple=lambda row:"/".join(f"{row[f'top_{count}']['median_fidelity']:.3f}" for count in (32,128,256))
        lines.append(f"| {run['seed']} | {err['maximum_state_relative_error']:.2e}/{err['logit_relative_error']:.2e} | {triple(mean)} | {triple(roll)} | {mean['contains_bilinear']['signed_projection']:.3f}/{roll['contains_bilinear']['signed_projection']:.3f} |")
    lines += ["","Passing this red team establishes numerical completeness for the declared finite path expansion. It does not make individual paths unique semantic objects: legal regroupings and cancellations remain, and rankings are specific to the identity-correction intervention family."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",OUTPUT,REPORT,flush=True)


if __name__ == "__main__": main()
