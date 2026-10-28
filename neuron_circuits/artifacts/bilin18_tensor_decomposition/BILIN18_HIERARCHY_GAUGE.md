## H697 — Hierarchical units have a reconstructively invisible coordinate freedom

Construct an explicit decoder mixing D'=D T, where T is invertible with identity diagonal blocks and ancestor-row/descendant-column off-diagonal blocks. Transform coordinates by z'=T^{-1}z. Reconstruction is unchanged because D'z'=Dz. On hierarchy-valid supports, the coordinate transformation introduces activity only into ancestors of active descendants, so it preserves allowed support. A child-only support without ancestors violates that condition.

Test all128seed patterns of H696's seven-node tree,32Gaussian samples per pattern, in double precision. Use both the orthogonal teacher B and a nonorthogonal teacher B S whose children intentionally include0.8times a parent direction. The hierarchy and transforms are supplied; this is a constructive identifiability audit, not training or discovery.

Both teachers retain reconstruction to relative error below8e-32 after mixing, but component-removal effects change substantially (relative squared effect errors0.619–3.230). All valid-support leakage is exactly zero. A deliberately invalid child-only mask gives reconstruction error2.14 after masking the compensating ancestor coordinates. Thus the ambiguity depends on the activation implications, rather than being an unrestricted claim about every independently gated representation.

### An ancestor-residualized convention

Project each node's decoder directions off the span of its ancestors, in topological order. Track the corresponding invertible triangular basis change and apply its inverse to the already-gated coordinates. This preserves total reconstruction, and the hierarchy ensures transformed coordinates remain on valid support. This is a posthoc transformation of gated codes; it is not automatically equivalent to replacing the original raw linear encoder with another raw linear encoder.

| Teacher | Reconstruction error after convention | Largest component-effect error after convention |
|---|---:|---:|
|orthogonal|7.26e-32|1.37e-14|
|nonorthogonal|7.51e-32|1.11|

The convention recovers orthogonal teacher contributions to numerical precision. For the intentionally nonorthogonal teacher, leaf effect errors remain0.390 and root error1.114 despite unchanged reconstruction. The algorithm has reassigned shared ancestor content; it has not discovered the original nonorthogonal generative decomposition. Orthogonality is an assumption/convention, not a consequence of reconstruction.

Applying the same procedure to the three H696 learned fits preserves reconstruction with relative change below5e-32 and reduces non-root effect errors to at most1.22e-10. A small root discrepancy remains if decoder bias is kept separate: relative error0.00135–0.00592. Assigning that fixed bias to the always-active root on H696's nonempty support distribution reduces root effect error below6.7e-12. This is a second explicit attribution convention. Removing the root plus its assigned bias differs from the original intervention that held decoder bias fixed; these must not be conflated. Total reconstruction metrics include decoder bias throughout.

This supports the user's data-based canonical-form direction with a necessary qualification: observationally equivalent decompositions can be represented by a chosen convention, while internal component interventions remain convention-dependent. To claim actual reusable computational units, we need evidence beyond reconstruction, including constraints from model interventions, transfers and robustness to representation choices. Neither the residualized synthetic fit nor this invariance audit establishes a unique unit of LLM computation.

Audits: all128seed patterns tested; valid support preserved; invalid-support negative control fails as predicted; source hashes unchanged, finite results. No original-model forwards/new documents; reserved LLM test split untouched. The trained fits are evaluated in double precision from saved float32 weights, accounting for minor numerical differences from H696.

Next broaden to computational DAGs: test whether learned sparse edges recover selective shared computation under heldout combinations, with redundant-path controls and explicit node/edge interventions. Retain the hierarchy convention as a candidate analysis tool, then evaluate representations on early LLM activations under stability, geometry and causal criteria. Joint unsupervised discovery and fair-budget comparisons remain incomplete. Broad goal active.
