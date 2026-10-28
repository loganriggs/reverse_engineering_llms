## H433 — Frozen factor-group recovery depends on output context

H432 was progress: output-state swaps separated receiving-context sensitivity from special tail-origin response directions. H433 returns to frozen computational units: H410 core64, added64 and union128 factors. For each H432 source/origin pair, compute native amplitude1 coefficient changes for each group without refitting. Insert full or group-only response increments into each of17 output states, preserving the baseline state. Compare full-target versus group-prediction exact KL and baseline Fisher residual energy. This measures reconstruction of an intervention response, not necessity of the group in the intact model.

The design retains48 frozen natural/Gaussian source vectors,17 origins and17 destinations. Common-length controls scale every group by the same full-response scalar, preserving the core+added=union identity. Recovery is1 minus pooled prediction error divided by pooled full effect. Negative recovery means worse than retaining the intact output, not negative accuracy. The tail case and sources remain post-selected; control/control averages all256 origin/destination pairs. No fresh documents or confirmatory population claims; total11456.

| Full-response length | Origin | Destination | Core64 KL recovery | Added64 KL recovery | Union128 KL recovery |
|---|---|---|---:|---:|---:|
|Original|Tail|Tail|83.84%|1.57%|90.32%|
|Original|Tail|Ordinary|72.16%|51.17%|95.36%|
|Original|Ordinary|Tail|91.59%|19.25%|95.14%|
|Original|Ordinary|Ordinary|81.77%|-27.46%|82.30%|
|Common|Tail|Tail|82.46%|6.56%|86.88%|
|Common|Tail|Ordinary|68.01%|51.86%|90.87%|
|Common|Ordinary|Tail|86.29%|17.66%|90.82%|
|Common|Ordinary|Ordinary|76.24%|-10.72%|80.05%|

Raw vector-energy recovery for union128 is97.97% for tail-origin and97.39% for ordinary-origin responses. That similarity does not carry over to output-distribution recovery. For ordinary-origin responses, moving the output destination from ordinary states to the tail raises union KL recovery from82.30% to95.14%; with common response lengths, from80.05% to90.82%. For tail-origin responses the destination change goes the other way: union recovery is95.36% at ordinary outputs and90.32% at the tail (common-length90.87% versus86.88%). Thus a universally more sensitive output state does not simply make every group better or worse: response origin and destination interact.

Added64 alone is useful for tail-origin responses at ordinary outputs (51.17% recovery) but harmful for ordinary-origin responses at ordinary outputs (-27.46%). The same qualitative reversal persists under common full-response lengths (51.86% versus-10.72%). Combining added64 with core64 improves recovery in all eight tested cells, but by very different amounts. An isolated group’s score therefore does not establish a stable computational role; cancellation/complementarity and the receiving output metric matter.

Natural and Gaussian families show similar original-length union results: ordinary/ordinary82.24% versus82.38%; ordinary/tail95.19% versus95.08%; tail/ordinary95.02% versus95.72%; tail/tail90.78% versus89.63%. These descriptive similarities do not establish equivalence. Common-length family differences are larger in some cells, so source-distribution conclusions still need uncertainty estimates and independent confirmation.

High fractional recovery also does not mean uniformly small absolute error. Ordinary-origin/tail-destination effects are much larger than ordinary/ordinary effects in H432. The recovered fraction and residual absolute KL answer different questions. Neither the high union scores here nor97% raw-energy recovery licenses a model-wide compact ontology or semantic label.

Numerical audits pass: core+added versus union response relative error 3.44e-07 below1e-5, full-effect replay against H432 exactly0.0 below1e-6. All saved scores are finite. Frozen set and response/state hashes are checked; groups are not reselected. H429 statistical failures are unchanged and no contexts are excluded.

Current interpretation: native factors provide a compact partial reconstruction, but stable membership alone does not imply stable causal roles across output contexts. The main open question is whether a representation that includes the output-sensitive combinations transfers better than individually ranked factors. Before designing such a representation around this case, independently test the observed context dependence.

Next: select receiving contexts by baseline information alone (for example low versus ordinary output RMS) on a disjoint panel, and freeze sources and response-origin comparisons before looking at group recoveries. Test whether union coverage and the sign of added-group usefulness change with destination, and whether output probability geometry predicts those changes beyond RMS scale. Use pooled finite-KL reconstruction, absolute residual KL and raw-vector recovery together; include matched-context and crossed-context conditions. A failure to generalize would constrain this tail-case interpretation rather than justify tuning on it.
