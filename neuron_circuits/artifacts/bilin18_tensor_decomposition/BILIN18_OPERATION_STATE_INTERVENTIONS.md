## H749 — Compact edits transfer from surgical edges to actual query/key states

Previous experiments prescribed a single attention-edge coefficient. This stronger test modifies L13H0's actual second query at the final destination and/or second key at the source, before normalization and rotary position encoding. The edit preserves the recipient raw projection norm and implements the frozen normalized-state target. All incident attention edges and both local and layer-zero bus values respond naturally; later layers run normally. Key edits can also change earlier query outputs that attend to the source. Other heads and the first attention factor receive no direct edit.

The full reference exchanges donor query/key states while restoring recipient norms. Compact variants use the original H742 or pooled H746 rank-four subspaces separately for query and key. Identity, wording and random rank-four controls receive identical treatment. Joint, query-only and key-only edits each have their own mode-matched full reference. All240 H747 contexts are evaluated, including120 diagnostic list contexts; nothing is fitted on this panel.

| Intervention on competent narratives | Original operation margin error | Pooled operation margin error | Identity control | Wording control | Random control |
|---|---:|---:|---:|---:|---:|
| Query only | 23.4% | 26.1% | 99.0% | 86.2% | 103.3% |
| Key only | 20.9% | 31.8% | 74.0% | 91.4% | 99.9% |
| Joint | 16.8% | 19.3% | 88.0% | 91.3% | 101.8% |

Margin error is RMS difference between compact and full intended-answer-margin changes, divided by RMS full change. Full changes have RMS0.278 query-only,0.250 key-only,0.510 joint logits. Original joint error is16.5%,11.1%,17.3%,18.2% across old-name/A,old-name/B,new-name/A,new-name/B narrative cells. These are reused H747 names and wordings, not a new holdout after H747.

Original joint head-output-change error is17.0% versus22.5% pooled,93.2% identity,91.9% wording,104.3% random. The mean KL from the full-intervention output distribution to the original compact output is3.7% of the mean KL from the full output to native (denominator0.005919 nats). Corresponding ratios are6.0% pooled,88.1% identity,77.4% wording,103.5% random. This is a ratio of KL divergences, not a percentage of information recovered, CE preservation, or accuracy improvement. The original query/key-only KL ratios are8.3%/2.7%.

Native intended top-1 on these contexts is90.8%; original joint is90.0%, full joint89.2%. These interventions change a component of computation without reliably switching the model's overall task. Approximation of the full intervention is the measured target; the full intervention is not assumed beneficial.

Diagnostic lists remain substantially harder: original joint margin errors46.1–51.5% across the four cells. All fail native binding competence, and they are neither pooled into the competent table nor discarded. The improved narrative result cannot be generalized to all task families.

Interpretation: the compact component survives a more realistic intervention at actual attention states, where its effects are no longer constrained to a single edge or bus-value contribution. Its advantage over nuisance and random directions supports causal specificity. This is still a supervised paired-edit component at one head. We have not established an unsupervised feature, manifold topology, necessity, selective influence across unrelated tasks, or the upstream circuit that creates it.

Audits:5,040 model forwards; all input/source hashes unchanged; recipient raw-norm relative error1.98e-7; actual normalized-state relative error1.49e-7; native-logit replay maximum1.24e-5; native-mode replay8.58e-6; all outputs finite. Native KL values near zero can have tiny signed floating-point errors. Zero original reserved-test evaluations; natural document count26,112 unchanged.

Next: trace which earlier layer contributions create these query/key coordinates, and test selective necessity against matched nuisance directions. Use this verified causal target to evaluate whether unsupervised featurizers recover the component, rather than optimizing only activation reconstruction. Keep independent task and wording validation in that comparison.
