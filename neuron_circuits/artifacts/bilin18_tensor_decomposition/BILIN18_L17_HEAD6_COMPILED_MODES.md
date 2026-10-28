

## H341 — Which compiled output modes carry head6 amplification?

Reuse H339 head6 recipients and all six moment/donor payloads. Compute a16×16 Fisher metric for the fixed compiler output rows at each intact recipient state. Multiply by the exact compiler coefficient differences to get the full mode-response Gram. Sum rows to allocate cross terms equally between modes. Rank modes by the average moment/donor fraction of excess energy relative to actual; select four on the same panel for exploratory localization. This is not heldout mode selection or a unique semantic decomposition.

Selected mode IDs (zero-based): **0, 2, 5, 1**.

|Condition|Total compiled sensitivity energy|Diagonal energy|Cross energy|Top4 retained fidelity|Top4 excess allocation|Top4-only excess energy fraction|
|---|---:|---:|---:|---:|---:|---:|
|actual|1.8374|2.0748|-0.2374|86.06%|—|—|
|moment|8.7319|5.4224|3.3095|89.61%|83.81%|77.06%|
|donor|9.3549|5.7742|3.5807|89.17%|82.62%|75.31%|

Retained fidelity=1−energy(omitted-mode sum)/energy(full compiled response). Excess allocation splits retained–omitted interactions equally; top4-only excess measures the retained Gram block directly. Negative allocations and values above100% are possible because cross terms are signed. None is a causal percentage.

Locked gates: top4_excess80: PASS, top4_fidelity80: PASS. Both controls must exceed80% for selected top4 excess allocation or retained fidelity respectively. These are descriptive concentration gates after within-panel mode selection.

**Limits.** Output modes are coordinates of an existing fixed compiler and may rotate under other valid representations. Concentration in these coordinates does not identify a semantic feature or show causal necessity. The full native remaining response, direct path and finite distribution change are not replaced here. Fisher geometry is local; downstream finite effects need separate testing.

**Audit.** Coefficient-output algebra relative error7.78e-07; per-document compiled Fisher energy replays H339 within1.41e-06; norm matching error6.97e-07. Frozen hashes verified; full16×16 per-document Grams, per-site coefficients, output metric and selected ranking saved. Native weights unchanged.

**Next discriminator.** Freeze the selected modes, then separate their compiler coefficient responses into linear and quadratic input contributions. Determine whether amplification comes from a mean-direction interaction or genuinely context-dependent terms before semantic interpretation. Broad research goal active.
