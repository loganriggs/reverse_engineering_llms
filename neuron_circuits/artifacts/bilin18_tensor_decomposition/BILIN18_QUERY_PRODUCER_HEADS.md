## H575 — Producer heads have distinct and opposing effects on query construction

On all 128 old-template edited binding/copying cases from H572, resolve attention layers 4, 7 and 9 into their nine head contributions. Capture the concatenated pre-output-projection head outputs. Reconstruct the layer write as the sum of each head slice multiplied by its output-projection slice. Clamp one final-position head slice at a time to its base-prompt counterpart; other heads and all later computation remain live. Include whole attention-layer clamps, MLP10 and self. These 32 modes give 4,096 interventions.

A head slice includes its read from all source positions and both local/persistent value components. This is not a single-source intervention or a neuron-level feature. The measured target is L14H4's source-edge coefficient, with a separate full-output margin measurement. Phase residual ratio means remaining absolute difference from the position-only gate prediction relative to native; lower is not necessarily better task behavior.

| Discovery condition | Lowest-ratio individual head | Residual ratio | Highest-ratio head | Residual ratio |
|---|---|---:|---|---:|
|binding_old_paren|head_9_7|0.916|head_9_4|1.030|
|binding_old_comma|head_9_7|0.926|head_9_4|1.026|
|copying_old_paren|head_7_8|0.752|head_9_6|1.132|
|copying_old_comma|head_7_8|0.725|head_7_6|1.109|

Copying is especially sensitive to L7H8 and L4H0; binding is more diffusely supported by L9 heads. L9H6 has a task-dependent sign in this diagnostic: its clamp reduces the binding correction but increases copying discrepancy. This is incompatible with treating every head in a selected layer as contributing in the same way. Single-head effects need not sum to the whole-layer effect because subsequent computation changes.

The fixed selection rule is the lowest mean residual ratio across both delimiters, separately per task. It selects L9H7/L9H6 for binding and L7H8/L4H0 for copying. These selections are discovery findings; no independent transfer claim is made here.

Audits: summed head writes reconstruct attention outputs to relative error <=2.50e−7; whole-layer/MLP control margin replay matches H573 exactly; raw-query reconstruction <=2.13e−6; self margin <=9.06e−6 and coefficient <=9.26e−7; all finite, source unchanged. No new prompts or corpus documents. Next H576 freezes these head choices and tests individual/group clamps in the other templates, with parent-layer controls.
