## H781 — Upstream site swaps reveal distributed construction of contextual source values

Previous goal turn was progress: H780 identified context dependence at the unchanged copied-value words. H781 tests where upstream outputs can supply that dependence to L5H5, while excluding unrelated downstream bypass effects from the measurement.

For each attention and MLP module in layers0–4, replace its output at both copied-value positions with the matched donor output in a receiver-context run. Capture the resulting L5H5 local values. Then start a fresh, untouched receiver run and inject only the captured local-value difference, weighted by(1−lambda) and native receiver routing, into L5H5's final query output. The second pass deliberately removes other downstream consequences of the first-pass intervention. Compare against the complete donor local-value-word target from H780 and a zero edit.

All ten sites are screened, with reciprocal task changes, nuisance key renames and identity controls across the lexical cohorts. The same480aligned edits and32nuisance exclusions remain. This is a sufficiency-style path diagnostic, not whole-model necessity or a unique mediation estimate. The substituted output can propagate through multiple subsequent upstream paths before its L5 value readout is measured.

| Upstream site | Task absolute effect error to target | Nuisance absolute effect error to target |
|---|---:|---:|
| L0 attention | .2704 | .1545 |
| L0 MLP | .1597 | .1545 |
| L1 attention | .2244 | .1570 |
| L1 MLP | .1047 | .1351 |
| L2 attention | .2536 | .1673 |
| L2 MLP | .1454 | .1456 |
| L3 attention | .2869 | .1488 |
| L3 MLP | .1755 | .1145 |
| L4 attention | .2448 | .1337 |
| L4 MLP | .1745 | .0856 |

These are pooled absolute RMS answer-margin errors. Target RMS is.2692 for task edits and.1723 for nuisance edits. Nuisance relative ratios remain suppressed under the.25threshold. Per-cohort results are retained; this table is a descriptive site screen, not evidence of an invariant ordering across examples.

L1 MLP is the closest single-site task substitute, with38.9% relative effect error and39.8% relative readout-vector error. For nuisance key renaming, later MLP substitutions are closer, with L4 MLP the best single-site margin match in this screen. Its remaining absolute error.0856 is substantial against target RMS.1723, and its readout-vector relative error remains63.2%. No single module reproduces the complete contextual source-value change.

This favors investigating distributed construction and interaction rather than immediately naming a single “binding module.” The task and nuisance rankings differ, which is compatible with partially different computations for value identity and key/context dependence, but does not establish a clean decomposition. Attention can be causally upstream of MLP effects even when its isolated donor substitution is a weaker match; these results cannot be read as evidence that attention is unimportant.

Task edits also change input embeddings and may affect later computation through x0 reinjection and paths outside the substituted site. Those contributions are deliberately not replaced by a single module-output swap. Thus incomplete task matching is expected under some distributed mechanisms and is not a proof of insufficient representational capacity. Individual site effects must not be summed as independent contributions.

Audits:10,944 model forwards. Native logits, captured local values, actual donor-output replacements, first-pass identity local values, second-pass identity/zero logits and H780 target margins all replay exactly. Model/input/source hashes unchanged and all metrics finite. Captures include all site outputs and final isolated readout patches for reproducibility. No original reserved-test use; natural document pool remains26,112.

Next: perform the complementary restoration test in donor contexts. Restore each site's receiver output at the same value-word positions, measure the surviving L5 local-value difference, and read it out through the same native receiver routing in a fresh receiver pass. Compare what is removed with what the donor substitution can supply, retaining all ten sites. This distinguishes a site that can inject a donor-like signal from one whose output is required for the complete isolated value-path effect. It remains path-specific; even a strong restoration effect would not prove whole-model task necessity or a unique circuit fraction.
