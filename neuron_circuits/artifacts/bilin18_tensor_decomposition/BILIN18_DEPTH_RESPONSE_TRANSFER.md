## H803 — Later query residual response transfers reverse the context-dependent behavioral gap

H802 showed that swapping the earliest divergent source/query attention responses had only modest effects. H803 crosses response identity more broadly at selected depth cuts. At each cut, replace the recipient's residual with its own zero-perturbation residual plus the other context's perturbed-minus-zero-perturbation residual response. Do this at the query only or all token positions. Match lexical item, format, query choice, and transplanted L2 source state. Test both zero and +.50 perturbation coefficients. Layers are zero-indexed.

This is a response transfer, not a full state swap: the recipient baseline, original input embeddings, and persistent L0 value bus remain. The hypotheses below were written into the configuration before the response-transfer runs, rather than selected after viewing their results.

### Hypothesis refresh: 20 distinct candidates

1. **high:** Later cross-token readers regenerate context dependence.
2. **high:** Final readout responds differently to the same residual response on different baselines.
3. **high:** Multiple token paths bypass the two L3 sites patched in H802.
4. **medium:** A later MLP transforms source content conditionally.
5. **medium:** Query-state geometry gates a shared source signal.
6. **medium:** RMS normalization accounts for a substantial part of baseline-conditioned readout.
7. **medium:** Unembedding competitors, rather than intended-logit change, drive the logprob contrast.
8. **medium:** Perturbation amplitude crosses a nonlinear response threshold.
9. **medium:** Context changes response direction more than its norm.
10. **medium:** Context changes response norm more than its direction.
11. **medium:** Residual reinjection of original embeddings regenerates conditioning after a cut.
12. **medium:** Persistent L0 value bus provides a parallel context route.
13. **medium:** Same-key repeated queries pool distinct reader behaviors.
14. **medium:** Finite hybrid interventions leave the native manifold and exaggerate differences.
15. **medium:** Lexical or formatting features explain effects without semantic key equality.
16. **low after H800:** L2 source-local state alone explains the contrast.
17. **low after H802:** The earliest L3 source/query attention differences explain most output contrast.
18. **low after H799:** Unequal injected value direction or magnitude explains context difference.
19. **open:** Shared dictionary atoms transfer content but require separate contextual gates.
20. **open:** A response-family representation is more faithful than static manifold labels.

### Discriminating results

Context gap means different-key minus same-key change in intended-token log probability caused by the value perturbation, averaged over query choices and both transplanted local states. Units are natural-log probability changes.

| Residual transfer depth | Site | Vertical-bar gap | Colon/newline gap |
|---|---|---:|---:|
| None | — | +.3862 | +.2172 |
| L3 | Query | +.3820 | +.2156 |
| L3 | All tokens | +.1871 | +.1700 |
| L8 | Query | -.4830 | -.1252 |
| L8 | All tokens | -.4833 | -.1307 |
| L13 | Query | -.3964 | -.1112 |
| L13 | All tokens | -.4360 | -.1131 |
| L17 | Query or all | -.3991 | -.1870 |

L8 query-only transfer reverses the gap in both formats. Descriptive item-bootstrap intervals for the transferred gaps are [-.6737,-.3165] and [-.1958,-.0633]. Their paired changes from no transfer are -.8692 [-1.1595,-.6000] and -.3424 [-.4928,-.2031]. Final L17 transfers also reverse the gap: intervals [-.5501,-.2564] and [-.2725,-.1116]. All-token and query-only final transfers produce exactly identical full-vocabulary logits, as expected when only the final query position feeds the output head.

At L3, query-only transfer is nearly inert on average, whereas all-token transfer reduces the gap by -.1991 [-.2798,-.1197] and -.04725 [-.06376,-.02987]. The all-token effect cannot be attributed uniquely to source2: it includes every affected later position and the local MLP transformations absent from H802's attention-output intervention.

For colon/newline L8 query transfer, the mean perturbation effect becomes -.0742 in different-key contexts and +.0510 in same-key contexts, reversing the original +.1057/-.1116 signs. In vertical bars both effects remain negative but their relative magnitudes reverse. Top-1 accuracy remains 100% throughout colon/newline; vertical-bar accuracy is 98.4% without transfer and ranges 96.1–98.4% across transfer arms. These hybrid interventions therefore are not universally behavior-preserving.

### Updated interpretation

The context dependence is carried in a transferable query response by L8. A hypothesis in which only the final recipient baseline determines the sign of the gap is inconsistent with these transfers: at L17 the baseline is retained, yet swapping responses reverses the gap. Baseline-dependent readout may still modulate magnitude, and these results are not unique causal shares or proof of donor-response equivalence.

The most useful next target is the transition between L3 and L8. Early all-token responses matter more than the early query response, while by L8 the query response alone carries a strong transferable contrast. This prioritizes cross-token readers and local MLP transformations in L4–L8. It does not prove a single critical layer, monotonic buildup, semantic key-equality gate, or minimal circuit. The chosen depth cuts bracket the observed transition but intermediate layers have not yet been tested.

Next test consecutive cuts L4–L7 to find where query transfer first changes the contrast, then distinguish attention versus MLP contributions around that boundary. Compare query-only against source/other-token response transfers and preserve paired zero-delta controls. Freeze the resulting mechanism before fresh lexical and diverse-task validation. A shared-content featurizer or computational DAG should be assessed against this response-transfer behavior, not merely a reconstructed activation or static semantic label.

### Verification and limits

3848 model forwards: 104 native local-state captures, 416 untransferred hybrids, and 3328 depth/site/delta transfer runs. All tracked hashes unchanged. Matched projected direction error zero; local residual clamp relative error 1.40e-8. Every zero-delta transfer exactly reproduces its untransferred full-vocabulary logits. H800 response reproduction error is at most 5.00e-6. Final query/all transfer logits agree exactly. All response values finite. Full captured residuals and logits are saved for independent follow-up.

The 16 vertical-bar and 10 colon/newline lexical items are reused; the two formats share words and repeated-key query variants duplicate prompts. Bootstrap intervals average local states and queries within item, are descriptive, and have no multiplicity correction. Residual response swaps may leave the native manifold. The original reserved test remains untouched; natural-document pool remains 26,112. No new featurizer, manifold, or complete computational DAG is validated here.
