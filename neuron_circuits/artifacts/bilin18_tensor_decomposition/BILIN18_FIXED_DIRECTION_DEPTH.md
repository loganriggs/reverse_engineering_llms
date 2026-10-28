## H828 — Context dependence emerges immediately downstream; apparent portability can be output filtering

Trace all13 H827 groups with the same52 recipients and frozen block/random directions. Run native and +/-0.1 interventions, recording final-position responses through every downstream block: raw residual, attention-normalized input, attention output, pre-MLP residual, normalized MLP input, MLP output and block residual. Also capture final normalization, pre-softcap logits and final logits. This is260 full-model forwards. It localizes propagation; it does not yet isolate module mediation.

### The first downstream block already changes directions

Median cross-recipient derivative cosine for the block direction:

| Stage | Edit after block0, next block1 | Edit after block1, next block2 |
|---|---:|---:|
| Identical injected residual direction |1.000|1.000|
| Attention normalized input |.994|.988|
| Attention output contribution |.474|.290|
| Residual entering MLP |.816|.854|
| MLP normalized input |.796|.844|
| MLP output contribution |.349|.433|
| Completed next-block residual |.475|.591|

Initial RMS normalization preserves direction closely in these comparisons. Attention output responses differ substantially across contexts, even with near-parallel normalized incoming perturbations. The MLP response also differs, and the completed first downstream block already has considerably reduced alignment. Random directions exhibit analogous first-block residual cosines .467/.602, so this is not specific to learned BSF directions.

Attention sees different recipient histories and native states; its response difference is therefore compatible with context-conditioned attention computation. MLPs also receive context-dependent baselines AND already differing perturbations, so the observed MLP cosine alone cannot separate those causes. Contribution cosines cannot be added or interpreted as percentages of the final context dependence. Norm changes can matter even where directions remain parallel.

### Dependence propagates through depth

For edits after block0, block-direction residual alignment goes1.000→.475 at block1, .354 at block2, .164 at block3, and .121 at block17. For edits after block1 it goes1.000→.591 at block2, .396 at block3, .243 at block4, and .182 at block17. Later behavior is nonmonotonic, with some alignment increases. Final normalized residual cosines are .063/.097; uniformly centered final-logit cosines .084/.067. The low final alignment is therefore not created solely by the output metric or softcap.

These are medians over dependent recipient pairs from5/8 groups, not universal layer profiles. Only one perturbation family and two early injection sites are covered. A causal response-crossing intervention at the first downstream block is needed to determine which changing components drive the effect.

### Correction to the portable-block interpretation

H827 highlighted residual0 block56 because its final common-metric response cosine was .997. The new trace shows that this does NOT reflect an invariant internal response:

- First downstream residual cosine .640, falling to .183 at block17.
- Final normalized residual cosine .071.
- Pre-softcap logits: uniform cosine .087, common probability-weighted cosine .019.
- Post-softcap logits: uniform cosine .086, common probability-weighted cosine .997.

The exceptional alignment appears specifically after the context-dependent output softcap under the common probability-weighted metric. It should not be promoted as an internal portable computational unit. This corrects the interpretation of the exploratory H827 exception while preserving its measured output result.

Verify the softcap mechanism analytically: for z_out=30*tanh(z_pre/30), predict the response as (1-tanh(z_pre/30)^2) times the pre-softcap derivative at each native recipient. Across all groups/directions/recipients, median relative prediction error is3.34e-5 under uniform centering and6.73e-5 under the common metric; maxima4.22e-4 and.00121. This supports saturation-dependent output filtering as the source of the apparent portability in that metric. It is not evidence of shared internal computation or broad semantic invariance.

### Audits and numerical limits

All saved final logits replay H827 exactly. Input/source hashes unchanged, finite metrics,260 whole-model forwards, no reserved-test evaluations or new documents; natural-document pool26,112. Source central derivatives match injected directions within2.30e-6 relative error.

The initially logged unparenthesized residual subtraction has maximum absolute error.0078125 because subtracting large float32 endpoints suffers rounding. A follow-up parenthesized native identity residual==(preMLP+MLP) is exact; native float64-summed relative discrepancy is3.46e-8. For saved central derivatives, additive relative discrepancy reaches1.156e-4 (0.0116%), narrowly exceeding an attempted1e-4 diagnostic threshold. Retain that failed strict diagnostic instead of claiming exact derivative additivity. This small endpoint-cancellation scale does not explain the observed cosine changes, but exact module-share attribution would require higher-precision or analytic response evaluation. Auxiliary additive and softcap audit JSON files are saved.

### Next discriminating intervention

Focus on the first downstream block, where alignment first drops. Cross baseline versus incoming perturbation at the attention and bilinear MLP separately, with RMS normalization explicit and recipient history retained. Test whether replacing a recipient's module response with a fixed-reference response restores cross-context alignment while preserving skip paths. Keep all groups and random controls, and evaluate downstream effects before interpreting mediation. Distinguish context-dependent attention reading, state-dependent bilinear MLP multiplication, and their interaction. The accumulating evidence favors a computational description conditioned on state over context-free feature directions, but a reusable operator or regime has not yet been identified.
