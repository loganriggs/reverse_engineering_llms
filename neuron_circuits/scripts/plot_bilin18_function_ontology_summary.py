#!/usr/bin/env python3
"""Plot the current variable/function/destination ontology results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


FIGURE=TENSOR_ROOT/"bilin18_function_ontology_summary.png"
REPORT=TENSOR_ROOT/"BILIN18_FUNCTION_ONTOLOGY_SUMMARY.md"


def cumulative(values):
    values=np.asarray(values,dtype=float);return np.cumsum(values)/values.sum()


def main():
    templates=json.loads((TENSOR_ROOT/"bilin18_bus_aware_function_templates.json").read_text())
    scales=json.loads((TENSOR_ROOT/"bilin18_full_operator_scale_sweep.json").read_text())
    fisher_files=[("L3H8", "bilin18_destination_fisher_geometry.json"),("L2H5","bilin18_destination_fisher_geometry_reverse.json"),("L4H1","bilin18_destination_fisher_geometry_third_target.json")]
    semantic=json.loads((TENSOR_ROOT/"bilin18_destination_fisher_mode_semantics_scattered.json").read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    axis=axes[0,0]
    axis.plot(np.arange(1,55),cumulative(templates["weight_rank"]["eigenvalues"]),label="raw bus-aware operators")
    axis.plot(np.arange(1,55),cumulative(templates["natural_rank_a"]["eigenvalues"]),label="natural responses")
    axis.axhline(.9,color="black",ls="--",lw=1);axis.set(xlabel="Number of modes",ylabel="Cumulative energy",title="Intrinsic function resolution");axis.grid(alpha=.2);axis.legend()
    axis=axes[0,1];all_scales=[.05,.2,.5,1.0]
    for name,label,color in [("L3H8_to_L2H5","L2H5 into L3H8","C0"),("L2H5_to_L3H8","L3H8 into L2H5","C1")]:
        for half,style in (("a","-"),("b","--")):
            values=[scales["results"][f"{name}_{half}_{scale}"]["partner_percentile"] for scale in all_scales]
            axis.plot(all_scales,values,marker="o",ls=style,color=color,label=f"{label}, {half}")
    axis.axhline(.1,color="black",ls=":",lw=1);axis.set(xlabel="Intervention scale",ylabel="Rank percentile (lower is better)",title="Directional operator compatibility",xscale="log");axis.grid(alpha=.2);axis.legend(fontsize=8)
    axis=axes[1,0]
    for label,filename in fisher_files:
        payload=json.loads((TENSOR_ROOT/filename).read_text());a=np.asarray(payload["rank_summaries"]["a"]["eigenvalues"]);b=np.asarray(payload["rank_summaries"]["b"]["eigenvalues"]);curve=(cumulative(a)+cumulative(b))/2
        axis.plot(np.arange(1,len(curve)+1),curve,label=label)
    axis.axhline(.9,color="black",ls="--",lw=1);axis.set(xlabel="Destination-Fisher modes",ylabel="Cumulative energy",title="Stable dominant mode, variable tail");axis.grid(alpha=.2);axis.legend()
    axis=axes[1,1];features=semantic["feature_enrichment"][:8];names=[row["feature"] for row in features][::-1];ratios=[row["energy_ratio"] for row in features][::-1]
    axis.barh(names,ratios,color="C2");axis.axvline(1,color="black",lw=1);axis.axvline(2,color="black",ls="--",lw=1);axis.set(xlabel="Mode-energy enrichment",title="Scattered FineWeb semantic audit");axis.grid(axis="x",alpha=.2)
    fig.suptitle("Bilin18: variables, functions, and destination-conditioned equivalence",fontsize=14)
    fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Function ontology summary","","![Function ontology summary](bilin18_function_ontology_summary.png)","","The panels separate four questions that should not share one monosemanticity score:","","1. Raw bus-aware operators have moderate linear compression (rank 22 for 90%), while natural head responses remain broad (rank 45).","2. Full-operator replacement is directional: L2H5 is uniquely compatible with later target L3H8 at every scale, while the reverse is ordinary.","3. Destination-Fisher spaces always have one dominant sensitivity direction, but the 90% tail varies substantially by target (5–27 modes).","4. The dominant L3H8 mode transfers to scattered FineWeb, but known token labels are only modestly enriched; it remains mechanistically clear and semantically unresolved.","","The visual is descriptive; exact gates, controls, and caveats remain in the linked experiment reports."]
    REPORT.write_text("\n".join(lines)+"\n");print(FIGURE);print(REPORT)


if __name__=="__main__":main()
