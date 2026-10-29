# Research wrap-up and toy-experiment handoff — 2026-10-29

Research in this thread is paused at the user's request. The user intends to pursue more toy experiments in a different directory. Do not restart experiments or the pending audit automatically. The broad research objective is incomplete.

## What the evidence currently supports

- Distributed implementation is distinct from removable redundancy. MLP1 tolerates substantial channel deletion on some binary task metrics, but quarter-channel deletion changes roughly 6–8% of top-1 outputs. That is not the same computation preserved exactly.
- Native MLP0–2 channel sums were net constructive on the matched data. MLP2 was less constructive, not established as cancellation-dominated. Cancellation between polynomial terms is a separate property from cancellation between native channels.
- Concentration in a chosen coordinate system, low-dimensional observed changes, low operator rank, and repeated native computations are different claims. Rotation and operator tests separated these properties; no single compression number establishes understanding.
- Hard top-k featurizer boundaries can create discontinuities absent from the smooth native MLP. M110–M114 established the mechanism, a continuous alternative, and native-input-path confirmation. The continuous alternative did not uniformly improve fidelity: it worsened natural reconstruction/output metrics, and much of its task-response improvement was explained by attenuation. This boundary detour is closed.
- Semantic blocks/manifolds and reusable native computational units remain unverified. Downstream readers can change the effect of a local signal; successful local reconstruction does not establish causal equivalence.

## Last completed, analyzed study: M115

See SENTIMENT_COMPOSITION_SCREEN.md. A crossed panel varied adjective valence, really/not, subject, and output format. Review-format semantic-group choices tracked all 64 examples, but a frozen full-vocabulary competence gate failed in one cell. Text-format choices stayed positive despite a measurable valence-by-modifier interaction. Both formats failed the frozen gates; neither was promoted to semantic training supervision. This is evidence that format/readout matters, not proof of an early negation circuit.

## Last run: M116 — saved, NOT independently audited or interpreted

The producer receipt reports completed_pending_audit: 1,920 forwards, 128 exact baseline replays, 1,792 intervention records, zero ineligible cases. No producer remains running at wrap-up. Cache, source, and protocol SHA256 values were checked against the receipt during wrap-up; this is an integrity check, not an independent scientific audit.

Purpose: transfer MLP1 lexical-change vectors between modifier contexts, formats, or both, at changed-input and final positions. Compare native vectors, transferred vectors, and norm-matched controls. Discriminate reusable lexical signals interpreted differently downstream from context-dependent early signal generation. Actual native responses are teachers; intended semantic labels are not.

Files:
- probe_composition_transfer.py
- validation/composition_transfer_protocol.json
- validation/composition_transfer_native.json
- Local large cache: /home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition/emlp_m116_composition_transfer.pt

Remaining work, only if explicitly resumed: independently reconstruct all deltas and norm controls, verify coverage, recompute full-output response errors and margin changes from saved logits, retain modifier/format/site cells, and report interpretation limits. Do not rerun the producer or infer results from its completion status.

## Useful starting contract for the separate toy experiments

Start with known mechanisms and test whether measurements distinguish them: disjoint cluster-specific units; distributed rotations of the same function; duplicated/substitutable contributions; shared-plus-private components; large opposing contributions with a small net write; and downstream-insensitive directions. These can coexist.

First compare exactly function-matched realizations to isolate representation effects. Separately train capacity-limited models to study which implementation optimization selects. An arbitrary change of basis alone cannot establish a learned strategy's capacity advantage. Match the function class, parameter/active-dimension budget, data, training budget, and declared metric where relevant.

Vary task complexity versus capacity, independent input noise, channel damage, context shifts, and unseen combinations. Keep local write error, output KL, CE, task accuracy, and intervention-response fidelity separate. Freeze comparisons before outcomes and distinguish predictions derived from assumptions from empirical observations. More useful toys produce discriminating joint signatures that can later be checked in the native model.

No new toy directory was created: its location and experiment design are left to the user's next work session.
