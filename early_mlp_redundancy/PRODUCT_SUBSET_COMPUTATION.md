# Native-product deletion: constructive distribution and partial substitution

M90 measured all 17 output spaces (the native MLP1 product write and 16 frozen learned decoder spans), three fixed rankings, and five retained sizes: 128, 512, 2048, 2304, and all 4608. All 255 cases pass independent audit. Rankings and the diagnostic rescaling coefficient use only the 4092 unique training positions. No new transformer forward, training, or held-out evaluation occurred.

## What the deletion test shows

Each channel contributes its bilinear product times its output direction. We rank channels by mean squared individual contribution norm in the relevant output space, using lower channel index to resolve ties. Two fixed uniform random permutations provide nested controls. The table reports relative **squared** local error of the retained dynamic sum against the full sum in the same space, not downstream KL or causal-response error. The affine projection offset is unchanged; separate full-affine denominator results are in JSON.

| Retained products, energy ranking | Native raw error | Native after scalar compensation | Learned spaces raw error range | Learned spaces compensated error range |
|---|---:|---:|---:|---:|
| 128 | 0.837 | 0.804 | 0.753–0.763 | 0.628–0.649 |
| 512 | 0.648 | 0.587 | 0.528–0.539 | 0.354–0.368 |
| 2048 | 0.257 | 0.222 | 0.150–0.157 | 0.084–0.087 |
| 2304 (half) | 0.216 | 0.188 | 0.119–0.125 | 0.067–0.070 |
| 4608 | 0 | 0 | 0 | 0 |

Compensation is one unconstrained least-squares scalar multiplying the retained sum, fitted on these same training positions. It is a distinct intervention, not raw removal. For the energy-ranked half, this scalar is 1.229 in the native space and 1.310–1.322 in learned spaces. A subset with only 128 products remains a poor local replacement under this rule, even when rescaled. That does not prove no other 128-product representation or selection can work.

Random-half raw errors are 0.347 and 0.357 in the native space, and 0.276–0.292 in learned spaces. Their fitted scalar corrections reduce errors to 0.284–0.292 and 0.110–0.121, respectively. Energy selection helps, but many contributions still matter; the representations do not simply expose a tiny removable native-product dictionary.

## Constructive addition, with a quantitative theory

Let v_i concatenate channel i's output contribution over the measured positions. Let F=sum_i v_i, E=||F||², and D=sum_i ||v_i||². Then E-D is the net signed cross term across different channels. D/E is 0.4008 for the native write and 0.1276–0.1308 for learned output spaces. Thus cross terms are net positive: the total sum has more energy than the sum of individual energies. This is constructive addition in aggregate. It does not rule out cancelling pairs or cancellation on individual examples, nor establish that channels implement the same function. These are uncentered output statistics; common background contributions may dominate task-relevant differences.

There is an exact prediction for uniform deletion, conditional on any fixed v_i. Keep k of N channels uniformly without replacement. Write p=k/N and q=k(k-1)/(N(N-1)). If S is their sum, then

    E_random[||S-F||²] / E = 1 - 2p + q + (p-q) D/E.

This follows because each diagonal term is included with probability p and each ordered off-diagonal term with probability q. It needs no independence assumption about the channel contributions. The scalar minimizing expected error over random subsets is

    alpha = p / [q + (p-q) D/E],
    expected relative error = 1 - p² / [q + (p-q) D/E].

For half retention, the raw-error prediction is approximately 0.3502 for native outputs and 0.2819–0.2827 for learned spaces. The two observed random subsets are consistent with that scale. Two samples do not establish a distribution or confidence interval. The common ensemble-optimal scalar is also distinct from the individually fitted scalar for each subset. The formulas were derived after inspecting results, not preregistered; exhaustive small signed-vector cases independently verify them.

This gives a sharper account than calling the MLP simply redundant: these output spaces show constructive distributed accumulation, useful energy concentration, and partial scalar-compensated substitutability. Exact removal invariance and causal substitutability remain unestablished.

## Relation to featurizers

M89 separated output-feature dimension from input-product count. M90 shows why that matters on the observed inputs: a 128-dimensional output space can still receive useful contributions from many more than 128 native products. Selecting 128 high-energy products is substantially worse than using all products. The correct architecture may need a product dictionary that is larger than the output-feature dictionary, but these results do not identify its optimal size or prove that native products are the best basis.

We should next transfer these exact frozen subsets and training scalars to held-out inputs, without re-ranking or refitting. This checks whether the apparent substitution is a training-distribution artifact. After that bounded local check, return to finite causal interventions; neither local error nor the random-deletion formula establishes preserved model behavior. Do not launch an architecture or optimizer sweep from these results.

## Audit and numerical history

The producer computes retained vectors directly. The auditor reconstructs products independently in torch and evaluates all metrics through their Gram matrix; learned spans use SVD instead of the producer's QR. It verifies 51 rankings, 17 signed-energy inventories, and all 255 cases at the frozen tolerances.

The first audit caught a tiny nonzero residual from reordering all channels in the full-retention control. That control now preserves the original full sum exactly. A second audit caught cancellation from subtracting two large Gram totals to compute a theoretically zero cross term; the auditor now sums the selected-by-omitted block directly. Both changes preserve the intervention and tolerance. Original producer evidence is archived locally and the repair history is retained. Final audit passed.

Evidence: [protocol](validation/product_subset_protocol.json), [all results](validation/product_subset_results.json), [independent audit](validation/product_subset_audit.json), [random-subset theory checks](validation/product_subset_random_theory.json), [numerical repair history](validation/product_subset_identity_fix.json).
