## H1021 — Head13.0 is a stable consumer of the dominant cached lexical payload

Previous turn was progress: H1020found cached slot0transfers across vocabulary while small learned directions do not. This turn isolates downstream consumers instead of confusing a cache slot reused across layers with one attention head.

For all32historical language rows, test three payloads: whole cached vector, slot0only,slot4only. Each payload has all-layers, seven single-layer and seven leave-one-layer-out arms over layers11–17. At each layer supply the recipient or edited cache explicitly and discard the returned bus; an edited cache does not persist accidentally into unselected later layers. Residual changes caused by an edited layer propagate normally downstream. No individual example filtering or evaluation-based reader selection; discovery top readers are reported alongside all historical evaluation arms.

| Payload/consumer | Discovery | Familiar/templateB | New languages,A | New languages,B |
|---|---:|---:|---:|---:|
| Whole-cache swap at layer13only: aligned share of all-layer cache effect |92.1%|92.6%|78.0%|78.9%|
| Slot0swap at layer13only: share of all-layer slot0effect |104.5%|105.0%|102.7%|102.5%|
| Loss when layer13excluded from slot0swap |105.8%|106.4%|103.6%|103.5%|
| Slot0read at13.0: recovery of full natural country-swap effect |63.2%|61.3%|57.3%|56.6%|

Head13.0denotes zero-based layer13,headslot0. Values above100%reflect weak opposing contributions at later layers, not super-complete recovery or a probability. Summing single-layer slot0effects reconstructs all-layer slot0effect to1.1–1.6%relative error. The corresponding whole-cache sum error4.8–9.7%shows more interaction across payloads/readers. Scores are signed aligned effect projections with explicit denominators, not percentages of all computation understood.

Slot4is less stable: discovery chooses layer15(52.4%single-consumer aligned share), while new-language examples favor layer14(47.3–51.8%) and layer15falls to22.0–25.3%. Template-transfer slot4has substantial opposing layer14effect(-22.9%) and positive layer11/15effects(50.9/59.9%). This is evidence against treating all cached slots as fixed universal circuits. No combined reader subset was selected on evaluation or adopted.

Novelty/scope: head13.0was already an explicit target in the H882multidomain screen. The new contribution is isolating its first-layer-cached-value input as a stable, large share of this task's effect, with a correctly localized cache intervention and complete-boundary denominator. It is not a new discovery that head13.0exists or is relevant. The native suffix and routing remain live, so this does not yet give a standalone executable model or identify a universal semantic feature.

Interpretation for featurizers: a reusable computational unit may be a port-to-head operation with a context-dependent read coefficient and a rich lexical payload, rather than a fixed low-dimensional residual direction. The stable head13.0path explains substantially more of this task than the prior final-residual subspace. Its128-coordinate cached input is a natural architectural group to compare with learned sparse groups, but dimensional costs and routing inputs must be charged. Overlap or hierarchy remains unsupported until multiple tasks demonstrably reuse the same computation.

Audits: unchanged original recipient cache after every row; returned bus verified identical to explicitly supplied tensor and then discarded; identity interventions exactly0effect. Base/full-cache/headslot prior replays within1.91e-6/2.86e-6/3.81e-6absolute. IndependentTorch single/omission attribution exact.1504suffix sequences, no newprefix/backwards; cached boundary states reused. Allfinite;no reservedBSFtest data;natural-document pool26112unchanged. No active process;H961pending;broad goal active.

Next decompose the head13.0cached contribution into its weight-defined value/output map and context-dependent attention coefficient, with exact replay of the head write. Test fixed versus live routing and finite downstream effects, preserving the secondary paths as explicit residual error. Compare reuse across tasks at this port-to-head operation before introducing a learned gate or richer BSF architecture. A token/content-general payload operator and a task-specific selector are distinct hypotheses and should be tested separately.
