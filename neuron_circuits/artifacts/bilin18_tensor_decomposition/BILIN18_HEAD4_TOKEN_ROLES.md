## H664 — Head4 residual error comes predominantly from input-token contributions

H663 localized much of month-succession approximation error to layer14 head4 (zero-based). H664 reconstructs that head's final-position write as a sum over token positions. For each position, multiply the two normalized rotary query–key dot products divided by head dimension128 to obtain the signed bilinear coefficient; multiply by the effective value vector and project through head4's output-projection columns. Effective values include the module's own value contribution and persistent value bus with the checkpoint's native mixing parameter. These coefficients are signed products, not softmax attention probabilities.

Tokenizer offsets partition every position into four roles: instruction text, Input label and intervening formatting, actual input content, and trailing answer suffix. Tokens with non-whitespace characters are required to belong wholly to one role; whitespace-only tokens are assigned by their starting offset. Token IDs are checked against the frozen manifest, input substrings against explicit item fields, and no position is dropped. Role sums are bookkeeping partitions, not assumed semantic units.

Fit separate role-wise copy→task mean shifts on the four second-wording discovery items. For each held-out example, compute each role's true residual beyond that mean. Add selected exact role residuals to the same existing layer14 prediction, leaving all other head/layer predictions fixed. Evaluate all16 subsets of the four roles under both predicted and exact boundary states, retaining all96 cases. Full-role repair must reproduce H663's head4 repair before interpretation. These are donor-informed diagnostics, not deployable predictors.

**Predicted boundary, KL(task||edited):**

| Family | No repair | Instruction only | Input-label only | Input content only | Suffix only | All head4 roles |
|---|---:|---:|---:|---:|---:|---:|
|addition|0.007949|0.007991|0.007953|0.007310|0.007937|0.007338|
|agreement|0.005285|0.005229|0.005261|0.005242|0.005281|0.005159|
|antonym|0.011206|0.011192|0.011205|0.009891|0.011171|0.009849|
|capital|0.017518|0.017641|0.017449|0.009953|0.017519|0.009977|
|plural|0.010843|0.010851|0.010783|0.010121|0.010712|0.009940|
|successor|0.086528|0.085965|0.086743|0.032417|0.086553|0.032080|

**Exact boundary:**

| Family | No repair | Input content only | All except input content | All head4 roles |
|---|---:|---:|---:|---:|
|addition|0.001671|0.001357|0.001686|0.001366|
|agreement|0.001706|0.001660|0.001646|0.001598|
|antonym|0.009779|0.007631|0.009704|0.007562|
|capital|0.013138|0.005811|0.013173|0.005810|
|plural|0.007369|0.005283|0.007308|0.005217|
|successor|0.081667|0.021882|0.081642|0.021852|

For month succession, input-content repair alone nearly reproduces the full-head repair:0.08653→0.03242 with predicted boundary, versus0.03208 for all roles. With exact boundary,0.08167→0.02188 versus0.02185 for all roles. Repairing every role except input content leaves0.08621 or0.08164 respectively, essentially retaining the mismatch. The discrepancy is therefore in the input-position contribution to this head's write, rather than principally in instruction or answer-suffix positions.

Capitals and antonyms show the same dominant input-role pattern; plurals also benefit mostly from input repair under exact boundary. Agreement's small head4 error is more distributed among instruction, label and input roles. Addition's head4 contribution is modest because H663 identified head7 as its stronger layer14 residual. Input-only repair can slightly outperform all-role repair due to error compensation, so these numbers are not additive causal shares.

This narrows the mechanism but does not establish successful semantic retrieval or month succession. An input-token write depends both on its effective value vector and on coefficients computed from context-sensitive queries/keys. A large residual could arise because a fixed mean ignores changing token values even when routing changes are simple, because routing itself is item-dependent, or both. The task prompts are often incorrect, and fidelity is to their output distribution rather than to a ground-truth algorithm.

Next factor the aligned input-position contribution into coefficient and projected-value terms. Verify exact token alignment between copy and task prompts, measure how much the effective value changes across contexts, and reconstruct the head/input write from those factors. Then compare simple discovery-only predictions such as a mean coefficient change applied to the receiver's actual input-token value with the current unconditional mean write. For single-token months this is a particularly direct test of whether the useful computational unit is a context-dependent scalar gate on an input-specific vector. Multi-token inputs must be handled explicitly rather than collapsed to a selected token. Exact factor swaps can distinguish missing routing from missing content, but any claim of a predictive unit requires evaluation without held-out donor factors.

Audits:3072 whole-model forwards (2 boundary conditions×16 role subsets×96 cases),1152 cached attention-stage evaluations/reconstructions. Position-write sums match prior head4 projected writes with maximum relative error4.81e-7. Role-residual sums match prior head4 residuals within7.06e-6 relative error. No-repair outputs reproduce exactly; all-role repair reproduces H663 head4-repair logits within8.11e-6. Runtime hashes unchanged; KL finite; tokenizer identity, content-span and unique-role assignment checks passed. All six families retained. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
