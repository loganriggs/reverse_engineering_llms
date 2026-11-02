# Weight-only output priors for uncued rule edits

The [tag-contrast result](WEIGHT_ONLY_TAG_CONTRAST.md) recovered a compact first-layer response from weights alone, but the tag named the desired intervention and its single-token change imposed an eight-dimensional rank bound. This experiment removes that convenience. In the six-frame [readable joint-rule language](SURFACE_MIXTURE_V2_RESULTS.md), a grammar-valid edit changes the *previous verb for the current subject*, altering the induction proposal at a later verb slot. Other edits change the previous place or current lexical cue. There is no dedicated mode tag. We ask whether a task-specific weight prior helps find the model's local response directions without fitting edit examples.

The two-layer, width-64 bilinear model was already trained with exact-oracle verb soft targets. Its induction edit fidelity is imperfect: the complete model's mean L1 error against the oracle effect is **.272/.285** on the two frame splits below. The analysis concerns what that trained model computes, not complete recovery of the DGP.

## Why the later bond

A separate stop check on the four-layer v1 readable model patched the **first** attention bond under valid endogenous edits. At its final query position, the complete 64-direction patch left induction effect L1 residual **.388**, compared with the complete model's effect L1 magnitude **.393**; patching the full sequence lowered the residual only to **.365**. The first-bond response is therefore mostly insufficient for that model's induction behavior, regardless of its local geometric rank. The [v2 mediation audit](SURFACE_MIXTURE_V2_RESULTS.md) similarly found first-bond full-patch induction effect TV **.023** versus complete-model **.181**. We test the **second** attention bond, whose full local patch mediates most of the induction and previous-place effects. The v1 stop-check data are [last-position](surface_mixture_models_v1_rmsnorm/edit_basis_layer0_last_baseline.json) and [full-sequence](surface_mixture_models_v1_rmsnorm/edit_basis_layer0_all_baseline.json).

## Data-free prior and evaluation

Let `O` be the final attention output matrix, `U` the unembedding, and `γ` the learned final RMSNorm gain. We compute the right singular vectors of

\[
 (U_{\mathrm{verbs}}-\overline U_{\mathrm{verbs}})\,\operatorname{diag}(\gamma)\,O.
\]

The first eight vectors form the **weight-output-verbs** basis of the 64-dimensional pre-output-projection bond. Centering the verb rows removes logit shifts common to every verb. This uses model weights and the known set of verb tokens, but no sampled text or causal edits. The RMSNorm denominator and its derivative depend on activations, so this is a *linearized output prior*, not an exact weight-only Jacobian of the full network. Controls use the output matrix's own right-singular vectors, analogous vectors for the **entire** vocabulary, activation PCA, and a basis fitted to a particular edit family's local response.

The model was trained on all six frames. The **featurizer fit** uses `lead/subject/at` frames and is tested on `when/in-place/during` frames from a different stream; the reverse split swaps the frame sets. Weight-derived bases use neither fit set. We patch the edited donor's projected **final-position** bond change into the original recipient and compare with a complete 64-direction patch at that same bond. The table is mean L1 distance of the restricted patch's probability change from the **full local patch**; lower is better.

| Rule edit / split | Fitted matching edit basis, 8 | Weight-output-verbs, 8 | Generic weight SVD, 8 | Whole-vocab output prior, 8 |
|---|---:|---:|---:|---:|
| Induction / forward, 171 edits | **.124** | **.167** | .230 | .274 |
| Induction / reverse, 149 edits | **.118** | **.193** | .256 | .274 |
| Previous place / forward, 431 edits | **.139** | .432 | .406 | .507 |
| Previous place / reverse, 377 edits | **.151** | .445 | .399 | .521 |
| Lexical cue / forward, 400 edits | .420 | **.398** | .633 | .727 |
| Lexical cue / reverse, 344 edits | .513 | **.351** | .671 | .699 |

For induction, the verb-specific prior closes about **46–59%** of the gap between generic weight SVD and the fitted edit basis, depending on frame split. For previous-place edits, it is worse than generic SVD, showing that a shared output vocabulary does not imply shared internal response directions. The apparent lexical improvement is only about *approximating the local patch*: the **full** second-bond patch itself differs from the complete lexical model effect by L1 **1.100/.977**, because the edit also travels through unpatched paths. Neither basis recovers the complete lexical computation.

For induction, the full local patch differs from the complete model effect by L1 **.079/.062**; the fitted eight-direction patch differs from the exact **oracle** effect by **.357/.359**, and the verb-weight prior by **.424/.450**. Both worsen oracle-effect fidelity relative to the complete model (**.272/.285**). Thus the prior improves local mediation relative to a generic weight basis without making the DGP rule more correct. This is a single trained seed and six deliberately restricted frames, not evidence of a universal output-based canonical form.

The next useful comparison should optimize a basis for **held-out causal-effect fidelity** at matched dimension and model KL, alongside a control that preserves output sensitivity but ignores the rule edit. A separate larger-vocabulary, less templated joint DGP is still needed for a realism claim. This result favors combining weight information with causal supervision rather than equating singular-value energy with a discovered rule.

Reproduction: [mediation script](eval_surface_v2_rule_mediation.py), [forward](surface_mixture_v2_soft_models/weight_output_mediation_layer1_forward.json), and [reverse](surface_mixture_v2_soft_models/weight_output_mediation_layer1_reverse.json). The script's edit-fitting pairs and held-out frames are independent, while the language model itself has seen every frame during training.
