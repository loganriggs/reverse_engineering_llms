# One problem: the extra contribution of MLP1

**Status amendment:** the [history audit](HISTORY_AUDIT.md) found related completed experiments.
The factorial below is now a possible protocol bridge, not the default next run or an untested
mechanistic idea. First resolve native-versus-deployed parent differences and specify the missing
contrast it would test. The detailed design is retained for review.

## Target

Explain why MLP1 repeats a readily token-predictable signal while removing it can have a large
behavioral cost. Identify whether its remaining context dependence is an important correction,
and whether that correction relies on MLP0. Keep MLP2 as a comparison, not a second main project.

## Competing explanations

1. **Parallel token computation:** MLP0 and MLP1 independently construct overlapping information
   from token embeddings. Predicts transfer of token-conditioned writes across contexts, including
   after changing MLP0, once recipient normalization is accounted for.
2. **Serial refinement:** MLP1 reads a transformed version of the MLP0 write. Shared decodability
   hides a genuine dependency. Predicts selective loss of the MLP1 correction when that input changes.
3. **Scale/conditioning:** MLP1 establishes norms or coordinates needed by later readers rather
   than adding much new semantic information. Predicts large zero/mean-ablation effects but smaller
   effects from direction- or norm-preserving controls. Norm matching is not itself a natural intervention.
4. **Distributed cancellation:** native channels combine large opposing terms. Predicts that
   subset retention is poor even when a learned mixed-coordinate replacement is useful.
5. **Downstream compensation:** separate pathways rescue individual removals. Predicts an
   interaction in joint interventions, but an interaction alone does not identify its mechanism.
6. **Metric/distribution artifact:** token tables preserve average CE yet miss KL, rare contexts,
   or targeted decisions. Predicts worse full-distribution or stratified performance than pooled CE.

## First fixed experiment, not an open-ended sweep

Use disjoint document cohorts for fitting, development/instrument checks, and final evaluation.
Fit only a token-conditioned mean write for MLP0 and MLP1, with a global mean comparator.
Freeze support counts and table coverage before evaluation. At uncovered token positions retain
the native write (hybrid replacement); report all-position and covered-position scores separately.
This does not perfectly isolate covered positions because earlier edits can affect later positions.

Run native, MLP0-table, MLP1-table, and both-table conditions. Add individual/joint mean-write
replacement controls and exact identity replay. All later computation is live. Keep x0 and the
first-layer value bus untouched and explicitly part of the intervention boundary.

Primary outcomes: KL(native || edited) and change in observed-next-token CE. Also report native-top1
agreement, token coverage, and local write error. Report document-level paired uncertainty.
Use the same frozen replacements for several competent behavioral contrasts (e.g. article choice,
number agreement, punctuation, lexical recall), keeping task labels out of the table fit.
First verify the native model performs each contrast; failures are not supervision.

For a damage metric D, report the interaction `D(both)-D(MLP0)-D(MLP1)+D(native)` without prematurely
calling its sign redundancy or synergy. Keep direction-specific examples and baseline stakes.
A replacement learned on native inputs can fail under the other edit because its input distribution
changes; distinguish this from an intrinsic inability to compose.

## Discriminating follow-up (only after the first result)

Define the contextual correction `c = native_MLP1_write - table[token]` on a recipient state.
Compare keeping, removing, and swapping c between same-token contexts. Separate MLP0 input effects
from attention1 input effects using matched interventions, recomputing RMSNorm. Exact corrections
observe native output, so this is an oracle localization test, not a learned replacement.
Use paired donor contexts and unrelated output contrasts to distinguish specific correction from
large nonspecific damage. A selective-reader claim must survive a finite intervention, not just
source decomposition—rung491/492 already showed why.

## Stopping criterion and shareable result

Stop after the fixed factorial and, if interpretable, one correction-swap experiment. Produce a
single effect table and a short verdict: shared token component sufficient under this boundary,
serial correction detected, broad scale/cancellation effect, or unresolved. Negative/ambiguous
results are legitimate. Do not widen ranks, add featurizers, or declare a manifold from this assay.
Any next project should be justified by a remaining distinguishable hypothesis.
