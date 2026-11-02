# Whole-rule-family holdout for a causal attention-bond basis

This experiment asks whether supervision on several known rules yields a
compact attention-bond subspace that transfers to **different rule families**.
The teacher is a frozen, two-layer width-64 bilinear transformer trained on a
toy narrative generator with competing verb and object mechanisms. We fit an
orthogonal basis at the last position of the **second attention block's
pre-output bond**. Its 64-dimensional activation enters the output matrix,
then the final residual and vocabulary readout. No language-model weights are
changed.

The fit and development panels contain eight complete edit families: verb
cue, lexical and previous-place edits; object adjective, cue and previous-
event edits; and two oracle-no-effect controls. **Verb induction and object
induction are absent from all basis fitting and checkpoint selection.** They
appear only in fresh test panels. The teacher itself saw both rules during
ordinary language training: this is a featurizer-supervision holdout, not
model-training transfer or unlabeled task discovery. Each edited prefix is
grammar-valid and has an exact generator next-token distribution. We remove
identical `(kind, original prefix, edited prefix)` triples across splits.

For each fit edit, capture its original final-block residual `h`, original
bond activation `z`, and edited bond activation `z'`. A full-width local
patch changes only that bond: `readout(h + W z') - readout(h + W z)`. A rank-k
basis `U` instead patches `z + U U^T (z'-z)`. We fit `U` with equal weight per
observed edit family, minimizing squared probability-effect error against the
full-width **local patch**, starting from edit-contrast PCA. Development
selects the training step. Baselines are contrast PCA, context activation
PCA, output-weight right-singular vectors, random directions, no patch, and
full width. The target is teacher-local causal fidelity; this is not a loss
on exact-oracle correctness or a human-legibility objective.

The rank-8 runs were exploratory. After inspecting them, we tried rank 16,
then opened **new test-generator seeds** for the rank-16 comparison. The table
uses only those fresh tests, 160 valid examples per held-out family per
teacher. It reports mean L1 difference in next-token *probability change*
from the full-width local patch; lower is better. The two teachers were
independently trained but share architecture and generator settings.

| Rank-16 basis | Teacher seed 0: verb induction | Seed 0: object induction | Teacher seed 1: verb induction | Seed 1: object induction |
|---|---:|---:|---:|---:|
| No patch | .283 | .212 | .343 | .208 |
| Edit-contrast PCA | .124 | .159 | .227 | .184 |
| Context PCA | .116 | .162 | .180 | .199 |
| Weight SVD | .090 | .164 | .121 | .186 |
| **Causal-effect fit** | **.080** | **.057** | **.088** | **.047** |
| Full 64 dimensions | 0 | 0 | 0 | 0 |

The optimized basis beats every listed compact comparator on all four
held-out outcomes, especially object induction. The same direction holds in
the initially inspected rank-8 panels, but rank 16 was chosen after those
results and should not be treated as a prespecified width. The local readout
replays the unmodified teacher within maximum probability error 1.2e-6.
Optimization uses a separate, non-induction development panel. The fresh
tests removed 1 and 0 duplicate triples across splits. These checks make an
implementation or exact-instance leak unlikely; they do not establish
training-seed confidence intervals or transfer to natural language.

The **full-width mediation ceiling is the key limitation**. On the fresh
panels, the full local patch differs from the teacher's *total* edit effect
by L1 .073/.156 for seed 0 verb/object induction and .022/.218 for seed 1.
Thus the final bond is a much better mediation site for verb induction than
for object induction. For object induction, other residual or earlier-layer
paths contribute substantially. Even the optimized rank-16 basis differs
from the total teacher effect by .184/.236 on the two object panels. Its
object-induction error against the **exact oracle** is .491/.535, while the
native teachers' errors are .366/.376. Preserving a teacher-local effect is
therefore not the same as recovering the correct object rule.

Two no-effect controls also remain necessary. For verb adjective edits, the
optimized patch's test effect differs from full local patch by .009/.009 L1.
For object previous-place edits it is .030/.011. These are small in absolute
terms, but the native teachers themselves have spurious no-effect responses,
so a successful local-preservation score should not be sold as oracle causal
specificity. Activation reconstruction is also a separate measure: the
optimized basis often has *worse* activation MSE than a PCA basis while
preserving output effects better.

The next experiment should move from this single last-position readout to a
multi-site or path-level mediation test, especially for object induction.
Keep the same whole-family fit holdout and fresh-test protocol, but first
measure full-width intervention ceilings at candidate earlier bonds and
all-position patches. Only after finding a site or path set that mediates
the native response should we optimize sparse groups or units for sufficiency,
necessity, and low spillover onto other rules. In parallel, the language
generator needs broader surface forms and distributional validation; this
small synthetic vocabulary does not establish FineWeb realism.

## Which final paths carry the missing object effect?

A separate site-selection panel tested **full-width**, uncompressed bond
patches at both attention layers. Patching the first attention bond at the
last position has verb/object induction error from the native effect
.316/.252 for teacher seed 0 and .346/.219 for seed 1. Patching it at **all**
positions still leaves .284/.242 and .347/.237. The second bond gives
.066/.149 and .023/.196 on these panels; all-position and last-position
patches agree at the final block, as expected for the last-token readout.
Thus an earlier attention-output bond alone does not repair the missing
object mediation. These panels selected the next mechanism question and are
not untouched evaluation data for a new fitted basis.

At the final readout, the original and edited forward passes provide two
inputs: the residual state entering the last block (`h`) and that block's
pre-output attention activation (`z`). Exchanging them separately gives a
2×2 cross-world intervention table. Recombining the edited `h` and `z`
replays the native endpoint exactly. Mean L1 effect magnitudes on independent
panels are:

| Teacher / edit | Native effect | Bond only | Residual only | Nonadditive interaction | Bond-only error from native |
|---|---:|---:|---:|---:|---:|
| Seed 0, verb induction | .339 | .289 | .069 | .024 | .069 |
| Seed 0, object induction | .267 | .207 | .170 | .070 | .172 |
| Seed 1, verb induction | .353 | .337 | .027 | .011 | .027 |
| Seed 1, object induction | .247 | .180 | .198 | .078 | .192 |

The effect magnitudes are L1 norms of probability *changes*, not additive
fractions; directions can cancel. Object induction is carried jointly by
the prior residual and final attention output, with a non-negligible
interaction. The next compact representation should cover **both** of these
inputs, then test whether it preserves the combined native effect and exact
oracle behavior. A cross-world path split is operational evidence, not a
unique attribution of an upstream computation.

Reproduce with [`eval_v3_rule_holdout_bond.py`](eval_v3_rule_holdout_bond.py).
The fresh-test records are
[`results/v3_rule_holdout_bond_rank16_fresh28001.json`](results/v3_rule_holdout_bond_rank16_fresh28001.json)
and
[`results/v3_rule_holdout_bond_rank16_modelseed1_fresh28011.json`](results/v3_rule_holdout_bond_rank16_modelseed1_fresh28011.json).
The rank-8 exploratory runs and first rank-16 inspected tests are also in
`results/v3_rule_holdout_bond_*.json`.
The mediation records are
[`results/v3_bond_mediation_ceiling_seed29001.json`](results/v3_bond_mediation_ceiling_seed29001.json),
[`results/v3_bond_mediation_ceiling_modelseed1_seed29011.json`](results/v3_bond_mediation_ceiling_modelseed1_seed29011.json),
[`results/v3_final_path_decomp_seed29101.json`](results/v3_final_path_decomp_seed29101.json),
and
[`results/v3_final_path_decomp_modelseed1_seed29111.json`](results/v3_final_path_decomp_modelseed1_seed29111.json).
