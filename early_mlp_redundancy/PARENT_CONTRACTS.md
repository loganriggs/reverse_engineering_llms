# Which computation did each experiment actually change?

Source-audited 2026-10-28. This corrects the preceding history audit's overly broad use of
“deployed parent.” That wording conflated the native checkpoint and a shipped approximation.

| Experiment | Baseline computation | Change | Relevant denominator |
|---|---|---|---|
| Historical token tables | Native `TT.GPT` loaded by `load_elriggs` | Block-write or MLP-write table, depending on script | CE cost or recovery relative to mean ablation; definitions differ |
| C512 MLP1 interchange / MLP2 compensation | Authority-hashed native `TT.GPT` | Replace only MLP0.Down with a rank512 affine map; explicit MLP2 omission in control arms | Contrast-specific effects, practical margins, native/C512 physical state-write crossings |
| Compiler-v2.1 | Restored, hashed frozen whole-model ship | Learned or oracle corrections at early sites; exact MLP2 alternative | Remaining teacher KL and gain against projected rank64 oracle |
| New packet inspector / response pilot | Pinned native checkpoint | Explicit MLP write interventions only | KL against recipient baseline, separately for each background |

## Source evidence

The [C512 base evaluator](evidence/prior/evaluate_mlp0_c512_mlp1_interchange_v1.py)
constructs `TT.GPT` directly, loads the authority's checkpoint with `strict=True`, and converts
it to float32. `install_candidate` replaces only `mlp.Down`; native Left/Right and the outer
Down_bias remain. `C512Down.forward` implements `(hidden @ right.T) @ left.T + intercept`.
Its rank512 is the factorization of the **4608-product-to-1152 write map**, not 512 newly learned
bilinear products. Do not confuse this C512 program with the learned MLP2 512-product program.

The [compensation evaluator](evidence/prior/evaluate_mlp0_c512_mlp2_compensation_v1.py),
despite its v1 filename, explicitly points to v2 authorities/results. It calls the base loader
and installs that proxy only at MLP0.Down. The [physical interface functions](evidence/prior/mlp0_c512_mlp2_compensation.py)
use native block2 mixing, attention and MLP; their suffix uses native blocks3–17.
“Deployed MLP2” in the historical narrative means MLP2 left active in this computation.

By contrast, [compiler-v2.1](evidence/prior/early_mlp_state_complete_compiler_v21_final.py)
imports `ship_error_attrib`, restores the frozen ship realization, and installs a
`StateCompleteCorrectionHook`. N is its uncorrected shipped state, not a synonym for native.
Its N/Q/O and N/E lattice therefore cannot be read as a native-module removal experiment.

The historical table loader fetched the same model repository without pinning the revision.
The current inspector pins revision and weights SHA256. This source audit does not retroactively
prove checkpoint identity for every historical table result.

[AST-located source lines](validation/parent_contract_source_audit.json) and PROVENANCE hashes
make these claims inspectable. No historical deployed model was executed by this audit.

## The gap this identifies

C512 attenuation is tied to errors made by a particular approximation, while the central
research question concerns meaningful native changes. A generic “MLP2 repairs early errors”
interpretation needs evidence beyond that one error distribution. The new native pilot directly
compares a context-derived MLP1 write change with norm-matched random directions, using a frozen
MLP2-write control that preserves the intact baseline. It probes response, not tableability or
full-model replacement quality.
