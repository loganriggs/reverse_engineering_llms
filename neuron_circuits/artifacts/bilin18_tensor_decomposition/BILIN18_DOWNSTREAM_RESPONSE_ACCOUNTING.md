## H877 — Residual context dependence accumulates across downstream MLP updates

After H876 established a causal contribution of final normalization scale, this experiment accounts for the remaining response difference between matched copying and capital-recall contexts. The early reader L8H3 has a distributed downstream response: several MLP updates contribute, and no single update accounts for most of the remaining mismatch.

### Exact accounting and scope

Replayed all 192 H873 contexts at three frozen readers with identity and the same positive/negative common headwrites: 1,728 full-model forwards. Capture final-token attention and MLP updates throughout the model. The final residual difference equals the sum of update differences, each multiplied by the product of subsequent residual-branch multipliers. The embedding anchor is unchanged and cancels. The selected reader's attention difference is independently checked against its projected injected headwrite.

Use the symmetric finite response (positive minus negative, divided by twice the common amplitude). Under H876's shared fixed final RMS and uncapped readout, projecting the update vectors gives an exact additive output-response decomposition. Canonically orient endpoints and average within lexical pair, separately for familiar A/B, competent C, and failed-capital D. These are the same reused countries and contexts, not fresh confirmation or new task evidence.

An update's gap-alignment fraction is its copy-minus-capital response difference dotted with the total difference, divided by the total difference's squared norm across pairs. Fractions sum to one; negative fractions represent cancellation and values above one are possible. They are not fractions of causal necessity or independently removable variance.

### Results on competent format C

|Reader|Output contrast|Total relative mismatch, shared scale without cap|Largest aligned component|Alignment fraction|
|---|---|---:|---|---:|
|L13H0|country|0.095|L17_mlp|0.484|
|L13H0|capital|0.097|L13_mlp|0.578|
|L14H4|country|0.079|L17_mlp|0.563|
|L14H4|capital|0.068|L17_mlp|0.496|
|L8H3|country|0.358|L17_mlp|0.197|
|L8H3|capital|0.271|L17_mlp|0.242|

For L8H3, MLP updates collectively align with 91.1% of the country-output mismatch and 103.6% of the capital-output mismatch; attention updates supply the remaining 8.9% and −3.6%. Familiar A/B gives MLP alignment 105.8% and 87.1%. This supports prioritizing MLP responses as the next intervention target. It does not imply attention is causally dispensable: attention changes can alter later MLP computations.

For early-reader country output in C, leading contributions are L17 MLP (19.7%), L16 MLP (14.9%), L15 MLP (12.5%), L9 MLP (12.1%) and L8 MLP (11.2%). For capital output they are L17 (24.2%), L15 (19.5%), L13 (17.3%), L10 (12.1%) and L14 (11.9%). The distributed pattern argues against assigning the discrepancy to a single new feature or one exceptional layer based on these data.

Late-reader residual mismatch is smaller and can have more concentrated contributions, but the dominant component differs by reader and output contrast. A universal final-MLP explanation is not supported.

### Verification

Final states replay H876 exactly. Maximum final-vector relative recomposition error is 1.22e-05; selected-output recomposition error is 4.98e-06. The projected direct write agrees within relative norm 2.2e-06. H876 relative-response errors reproduce within 3.15e-15. All source hashes remain unchanged, all outputs are finite, and no reserved test was evaluated. The document pool stays at 26,112.

The first execution rejected the direct-write check because an absolute tolerance of 0.001 was too tight for these large, amplitude-divided vectors (observed maximum 0.001734). The revised runner retains the absolute discrepancy and adds a relative-norm check below 0.0001; observed relative error is about 0.0000022. The full-output check remains below 0.001, with actual error about 0.000005. The final-state vector's maximum coordinate recomposition discrepancy is 0.0471, with relative norm about 0.0000122. Thus the audit is scale-aware rather than claiming bit-exact algebra through float32 operations. An incidental repeat of the initial run occurred after a missing shell `python` command; all successful reported results are from the final runner.

### Next causal test

For L8H3, freeze final-token MLP updates to each recipient's unedited native value during the common positive/negative headwrite, first L17 alone, then L15–17, then all L8–17. Include the complementary attention-update freeze and a joint attention-plus-MLP freeze as an exact skip-path control. Keep preceding computation and unselected updates live. Check identity replay and the joint-freeze prediction. Evaluate native readout and shared-scale readout separately, reporting absolute response size alongside relative mismatch so collapsed responses cannot appear to establish reusable computation.

These freezes test whether downstream updates causally generate the mismatch; the accounting above is not a unique mediation assignment. They remain diagnostic interventions on reused panels. A manifold or task-blind reusable unit still needs independent task transfer, stability and selective causal tests. The broad research goal remains active.
