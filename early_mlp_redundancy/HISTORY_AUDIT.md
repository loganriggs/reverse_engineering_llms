# Follow-up history audit: do not repeat the older composition experiments

The initial packet's proposed table factorial was too easy to read as an untested research direction.
A follow-up source audit found related completed experiments and a crucial continuation of the MLP2
compensation study. This amendment changes the next action; no new model experiment was run.

**Parent-contract correction:** subsequent source tracing established that the C512 compensation
assay uses the native checkpoint with only MLP0.Down replaced. “Deployed MLP2” there means
the native MLP2 left active, not the approximated whole-model ship. Compiler-v2.1 does use
the frozen ship. See [PARENT_CONTRACTS.md](PARENT_CONTRACTS.md).

## Related table compositions already exist

[Two-block composition](evidence/prior/front_table_compose_results.json) reports CE costs
.1666 and .5218 individually, .6654 jointly, and 1.0647 after sequential refitting.
The joint-minus-singles difference is -.0230 nats using the rounded report values.
But [the code](evidence/prior/front_table_compose.py) captures **attention + MLP writes**,
uses rank64 interfaces, and uses a current-token table for block0 versus a token-pair table for
block1. It is not the pure MLP0/MLP1 unigram-table factorial proposed in the initial packet.
Its basis construction also accesses the same `fresh` rows later scored by `price`; do not call
that a wholly untouched evaluation panel, even though table means use another corpus.

[Four-MLP composition](evidence/prior/front_table_compose4_results.json) tests pure MLP
unigram tables at sites0–3. Individual CE costs total 2.18904; joint cost is 2.52398, a ratio
of 1.15301. Its [code](evidence/prior/front_table_compose4.py) uses a position-weighted global
mean for unseen tokens, rather than the proposed hybrid native-write fallback. It does not report
all pair-specific factorial corners or output KL. These are concrete protocol differences, not
a reason to ignore the existing evidence or advertise another table fit as a new mechanism.

## The MLP2 follow-up limits the repair story

The [physical compensation v2 report](evidence/prior/MLP0_C512_MLP2_COMPENSATION_V2_FINDINGS.md)
follows the MLP1 interchange report already included. MLP2 presence and the ensuing suffix attenuate
the C512 mismatch descriptively, but the specifically aligned MLP2 write does not beat the shuffled
write control. Generic sensitivity is unpowered. The registered labels are inconclusive because
float32 coverage disagrees with exact ledger coverage by roughly 1e-8 under a 1e-12 equality gate.

The stored result confirms that all other common gates passed. That bookkeeping failure must not
be silently converted into a new positive result, nor treated as evidence against every mechanism.
The alignment failure remains substantively relevant even if the coverage comparison is repaired.

These runs concern a **C512 replacement of native MLP0.Down**, with the other checkpoint
components intact (apart from the explicit MLP2-omission control). Their results concern
approximation-induced errors, not arbitrary native MLP1 edits. Distinguish the upstream error,
local MLP2 write, and downstream response before comparing numbers.

## Later compilation found conditional utility, not a faithful replacement

The [compiler-v2.1 final report](evidence/prior/EARLY_MLP_STATE_COMPLETE_COMPILER_V21_FINAL_FINDINGS.md)
and its [terminal authority](evidence/prior/early_mlp_state_complete_compiler_v21_final_authority.json)
record a scientific negative. The executable pair recovers 26.10% of the rank64 oracle CE gain;
its remaining teacher-KL ratio is .66308, above the .50 admission ceiling. MLP1's compiled component
is harmful alone on the deployed parent but helpful after compiled MLP0. This supports ordered,
parent-dependent computation in that setting, not freely interchangeable redundant modules.

Exact MLP2 worsens both baseline and compiled arms there; the pair retaining a positive advantage
is not evidence that exact MLP2 repairs it. No universal “MLP2 compensation” claim is licensed.

## Revised next action

Before collecting a new natural-text panel, make the **parent/interface contract** explicit for
these three evidence streams: native token-table assay, C512-on-native compensation assay, and
compiler-v2.1. Identify which exact functions and persistent state differ. Use the historical
results to choose one missing contrast. The proposed native hybrid-table factorial can serve as a
controlled bridge, but should proceed only if it resolves that contrast; it is not the default
next experiment merely because it is easy to implement.

The most useful unresolved distinction is **specific information correction versus generic
attenuation under a changed operating state**. An assay needs an adequately powered matched
sensitivity control, native/deployed backgrounds kept separate, and new documents for any revised
inferential gate. Reusing spent rows can diagnose mechanics but cannot provide fresh confirmation.

[Audit receipt](validation/history_audit.json) records the arithmetic and stored decision checks.
This audit did not rerun bootstrap inference or validate every historical model execution.
