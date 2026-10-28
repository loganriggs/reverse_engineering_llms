## H876 — Direct final-scale clamp reduces late-reader context dependence, but leaves earlier computation effects

H875's parameter-free RMS ratio explained much of the failed format-gain transfer. H876 now intervenes on the final normalization operation while keeping all preceding computation live. For late readers in competent newformatC, a shared scale reduces cross-context response mismatch from10.8–22.1% to4.9–8.9%. This supports a causal role for the native scale difference. It does not eliminate downstream context dependence, especially forL8H3.

### Intervention definitions

Use all192 H873 endpoints and all3 frozen readers. Re-run identity and matched positive/negative headwrites, capturing the actual final residual state after all attention/MLP updates. Evaluate exact selected-token readouts on those same states under five denominator/readout choices:

- Native: RMS of the edited final state, as in the model.
- Own fixed: each context's unedited native RMS, held fixed under the headwrite. This removes perturbation-induced radial feedback but preserves cross-context scale differences.
- Shared dynamic: multiply edited RMS by shared/native RMS. The shared value is the geometric mean of matched copy/capital native RMS. Native baselines share a scale, but each retains its perturbation-induced radial feedback.
- Shared fixed: divide both contexts and both perturbation signs by that same native geometric mean. This removes baseline scale differences and perturbation-induced radial feedback.
- Shared fixed, no cap: same shared denominator, additionally omit the30tanh logit cap as a separate diagnostic.

All normalization interventions occur only at the output, so every preceding computation is live and identical across readout variants for a given headwrite. These are counterfactual output operations, not model retraining or a proposal to change its deployed normalization.

### New competent formatC: cross-context relative response RMSE

No gain is fitted. The copying response predicts the capital-context response for the same output contrast and matched headwrite. Responses are oriented to the same country contrast and averaged within each lexicalpair.

|Reader|Output contrast|Native|Own fixed|Shared dynamic|Shared fixed|Shared fixed, no cap|
|---|---|---:|---:|---:|---:|---:|
|L13H0|country|0.143|0.133|0.062|0.058|0.095|
|L13H0|capital|0.221|0.205|0.105|0.089|0.097|
|L14H4|country|0.152|0.153|0.050|0.049|0.079|
|L14H4|capital|0.108|0.102|0.083|0.077|0.068|
|L8H3|country|0.280|0.292|0.271|0.314|0.358|
|L8H3|capital|0.361|0.329|0.274|0.262|0.271|

The late-reader improvement comes primarily from equalizing native context scales: own-fixed alone gives no comparable reduction, while shared-dynamic already reaches5.0–10.5%. Removing perturbation-induced radial feedback as well gives4.9–8.9%. The reduction also holds in absolute response units, so it is not just an artifact of the relative-error denominator: for exampleL13H0 country-response absoluteRMSE falls.368→.144 andL14H4 country-response.424→.131.

For new pairs in familiarA/B formats, shared-fixed late-reader errors are7.2–13.8%, compared with13.7–30.9% native. FailedcapitalD remains a separately saved diagnostic. The clamp was motivated by prior results on these contexts, so this is a mechanistic intervention rather than fresh prospective generalization evidence.

L8H3 remains context-dependent. InC, shared-fixed country-response error increases28.0→31.4%; capital-response error decreases36.1→26.2% but remains large. Shared-dynamic yields27.1% and27.4%. Eliminating the logit cap is not uniformly helpful. Therefore neither baseline scale, radial feedback nor cap saturation alone explains all observed downstream variation.

### Interpretation for representation and computation

The static source direction and signed routing coefficient have a more reusable relationship to output once context scale is accounted for. This helps distinguish semantic/computational variation from a global normalization effect. It does not prove a unique canonical form, universal linear readout, or task-independent manifold. The same learned feature can have a different raw-margin impact because of normalization, and early-reader updates can additionally recruit different later computations.

The measured mismatch concerns causal response, not taskaccuracy or language-model loss. Equalizing scales is a diagnostic operation; the experiment does not show that it improves the model. Binding's earlier failed-country panel and competent human-name evidence remain separate, and broader task generalization is still outstanding.

### Next discriminating decomposition

Capture native and perturbed final-token attention/MLP updates after each frozen reader. The exact residual recurrence decomposes the final-state perturbation into the direct headwrite propagated through residual multipliers plus weighted differences in later attention andMLP updates, including the edited layer's MLP. Audit this vector sum against the actual final-state difference.

Under shared-fixed normalization with no logitcap, output response is linear in that final-state difference. This permits an exact accounting of which update differences explain remaining cross-context response mismatch. It is an accounting decomposition, not a unique causal mediation assignment: later updates interact and depend on earlier ones. Use it to select a concrete follow-up intervention rather than fitting another gain or assuming all residual error is a new semantic feature.

### Verification and scope

1,728 full-model forwards. Native logits and native final states replay exactly. Exact readout differs from model logits by at most7.54e-06; previouscommon-write margins replay within8.94e-06. Shared denominators are equal across matchedcontexts, and shared-dynamic identity error is0. H874 native response RMSE independently reproduces within2.32e-06. All source hashes unchanged and outputs finite; no reserved-test evaluations,26,112-document pool unchanged. Pair-bootstrap4096 uncorrected intervals are retained for12 reused lexicalcombinations, separatelyA/B,C,failedD. Broad goal remains active.
