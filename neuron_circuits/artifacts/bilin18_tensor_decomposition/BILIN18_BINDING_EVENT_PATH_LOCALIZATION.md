## H924 — Late attention writes carry the largest event-order contrast effects

H923 separated candidate-name promotion from the unexplained event-order preference. H924 localizes two matched contrasts across all18 layers, testing final-token attention and MLP outputs separately. The128 transfer rows are all retained regardless of task competence. Names and templates are familiar; this is localization on an existing panel, not fresh generalization evidence.

The event-swap donor toggles sender identity while holding list order, extra mention, wording and question fixed. Exactly the two event-name tokens change. The same-fact-reword donor changes sent-to into received-from or vice versa, preserving sender/recipient identities but reversing event argument order and changing the event wording. All pairs have equal token lengths, identical final tokens, and identical pre-event and post-event context. These contrasts separate changes to the fact from changes to its surface order, though the latter also changes lexical wording.

For each recipient, replace one native final-token attention or MLP output with its donor output. Other token positions, modules and downstream computation remain live. Self replacements at all36 sites use the first eight transfer rows as controls. Native writes are captured for all256 rows before intervention.

### Largest single-output effects

The reference margin compares the recipient's second event name against its first. Native mean margin is0.620, with94.53% binary preference for the second name. Full donor prompts reduce this margin by1.239 on average. Patch effects below measure reduction in this same fixed recipient margin; they are not task-accuracy improvements. Error-to-full compares patch effect with full-donor effect, relative to no patch.

|Contrast|Patched output|Mean margin drop|RMS change|Error-to-full ratio|Native→edited KL|Second name preferred|
|---|---|---:|---:|---:|---:|---:|
|event_swap|L14 attention|0.473|0.487|0.619|0.01220|58.59%|
|event_swap|L16 attention|0.217|0.226|0.825|0.00283|80.47%|
|event_swap|L13 attention|0.141|0.148|0.885|0.00118|86.72%|
|event_swap|L15 attention|0.135|0.138|0.894|0.00098|85.94%|
|event_swap|L17 attention|0.075|0.079|0.938|0.00038|91.41%|
|event_swap|L11 attention|0.065|0.071|0.948|0.00040|92.19%|
|same_fact_reword|L14 attention|0.473|0.483|0.621|0.01222|60.16%|
|same_fact_reword|L16 attention|0.217|0.221|0.828|0.00275|81.25%|
|same_fact_reword|L13 attention|0.141|0.150|0.883|0.00128|87.50%|
|same_fact_reword|L15 attention|0.135|0.138|0.892|0.00100|85.94%|
|same_fact_reword|L17 attention|0.075|0.080|0.939|0.00041|90.62%|
|same_fact_reword|L11 attention|0.066|0.078|0.946|0.00057|92.97%|

L14 attention is the largest site for both contrasts: mean drop0.473 logits, roughly38% of the full-donor mean change. Its event-swap patch reduces second-name preference to58.59%; same-fact rewording gives60.16%. L16 attention follows at0.217, then L13 and L15. The largest individual MLP RMS effect is0.028 for event swaps and0.043 for same-fact rewordings. Small final-token MLP patch effects do not exclude important MLP processing at earlier source positions or interactions among modules.

The two contrasts have similar aggregate localization but are not identical per example. L14 attention patch-effect correlation across contrasts is0.612, with RMS difference0.0935; L16 correlation is0.481 with RMS difference0.0562. The identical pooled full-donor mean drop is partly a consequence of the balanced donor permutations, not independent evidence that the model treats the contrasts identically. Individual records and contrast comparisons are saved.

### Interpretation and next experiment

The strongest final-token pathway for this shortcut lies in late attention writes in the current scan. This is a localization result, not a complete circuit: no single patch reproduces the full donor response, and effects cannot be summed as independent causal shares. Patching a donor write can also introduce context mismatch.

Next decompose all nine heads at L14 and L16 for both contrasts. Compare native donor head-output replacement with routing-only, mixed-value-only and persistent/current-value contributions. This distinguishes whether the difference originates in routing, which values are carried, or their interaction. Keep all heads rather than presuming H4 remains the selector: H923 found strong candidate support but incomplete shortcut explanation at that head. Any head selection from this scan is exploratory and will require transfer confirmation before a general mechanism claim.

The broader featurizer implication remains that reusable operation and reusable low-dimensional activation geometry are different hypotheses. We are locating the operation before choosing a representation to approximate it.

### Verification

9,760 forwards:256 native captures,9,216 donor substitutions and288 self controls. Native-cache replay and all self controls exactly zero. Token-alignment audits pass for both contrasts; source hashes unchanged and all measurements finite. Native module writes, all patch metrics and contrast-profile comparisons saved. No reserved-test evaluation or added natural documents; pool remains26,112. Broad goal active.
