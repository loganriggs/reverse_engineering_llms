# Preregistration: direction versus amplitude in the final grammar interface

The first red-team found that pure-rule semantic patches are substantially more
accurate than generic answer-class patches, yet their exact finite output effects
have cosine 0.992 and their top-32 native supports overlap by 0.676. This follow-up
tests whether the remaining difference is only amplitude.

Recreate the same bilinear checkpoints, generated direct-rule transitions,
prototype panels, identity split, and template split. For each transition:

1. match the generic answer delta's activation norm to the semantic delta;
2. on discovery-template examples only, choose one scalar from the frozen grid
   that makes the generic answer patch's mean new-versus-old margin change
   closest to the semantic patch's margin change;
3. freeze that scalar and evaluate exact finite effects on held-out identities
   using template 6.

Report target accuracy and softmax-Fisher fidelity to the semantic patch. The
scale is one scalar per declared answer transition; it cannot rotate the effect,
select examples, or use evaluation outcomes.

If discovery calibration closes the pure-rule gap, the final patch is best
described as a generic output verb with rule-conditioned amplitude. The rule
specific computation must then be sought upstream. Failure despite the already
high cosine would indicate meaningful nonlinear or example-specific structure
not captured by one scalar.
