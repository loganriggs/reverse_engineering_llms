# Recovering a causal rule-mixture broadcast from weights alone

The [conditional mixed-rule model](CONDITIONAL_MIXTURE.md) receives one of three visible tags specifying the sampling weights of a joint bigram/trigram/skip-bigram/induction generator. On a fixed prefix, changing the tag changes the exact next-token distribution at rule collisions. Earlier work found that an activation-contrast PCA basis, fitted on **no-conflict** streams, captures nearly all of the *first attention bond's* mediation of the model's tag effect using four of 64 coordinates. This experiment asks whether the same basis can be obtained without fitting any language samples.

## Weight-only construction and exactness check

At a non-tag query position `t`, the first attention layer's query is a function only of the current token embedding. Changing the tag at position zero leaves every other key and value unchanged. In head `h`, the donor-minus-recipient difference in the pre-output-projection attention result is therefore

\[
\Delta y_h(i,t)=\alpha_h^{(D)}(i,t)V_h e_D-\alpha_h^{(R)}(i,t)V_h e_R,
\]

where `i` is the current ordinary token, `e_R/e_D` are the tag embeddings, and each `alpha` is the **product** of the two RoPE-rotated QK scores divided by the squared head width. This identity uses the complete signed attention product; looking at one QK factor or one value vector in isolation would miss the composed effect. The formula bounds the full contrast span by two tag-value vectors per head, hence rank at most eight for four heads. It applies only to the first layer and non-tag positions. The trained checkpoint has no RMSNorm or QK normalization.

We enumerate all **26 ordinary tokens × 63 non-tag positions**, compute this contrast directly from embeddings and attention weights, and take its SVD under a uniform token-position metric. The SVD's left/right signs are arbitrary; its four-dimensional subspace is the proposed weight-only patch basis. The enumeration is a specified input-domain metric, **not** fitting to sampled text, a unique canonical gauge, or discovering which tag intervention to ask about. Against actual captured first-bond differences on 26 constant-token probe sequences, the formula's RMSE is **2.69×10⁻⁸** for baseline→induction-heavy and **2.71×10⁻⁸** for baseline→bigram-heavy. This verifies the weight calculation at every ordinary token and position.

## Held-out causal mediation

For comparison, activation PCA and tag-contrast PCA fit on 128 separately sampled baseline streams, using only ordinary **no-conflict** positions. Two independent evaluation panels use 256 and 512 newly generated baseline streams. Their visible bigram–induction conflicts number 796 and 1,713. For each fixed prefix, a donor run changes only the tag. We patch four coordinates of its pre-output-projection **first attention bond** into the baseline recipient, then measure RMSE from the donor-minus-recipient bigram-versus-induction log-odds shift. Lower is better; full-width patching establishes the floor for this bond, because the recipient retains unpatched tag information elsewhere.

| Tag change | Panel | No patch | Physical 4 | Weight SVD 4 | Data contrast PCA 4 | **Weight-only tag contrast 4** | Full-bond patch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline→induction-heavy | 7802 | 1.415 | 1.412 | 1.375 | .584 | **.584** | .581 |
| Baseline→induction-heavy | 7812 | 1.403 | 1.401 | 1.368 | .588 | **.588** | .585 |
| Baseline→bigram-heavy | 7802 | .887 | .887 | .839 | .414 | **.413** | .418 |
| Baseline→bigram-heavy | 7812 | .874 | .874 | .838 | .412 | **.411** | .417 |

The weight-only four-dimensional subspace matches the sampled contrast basis within **.001 RMSE** on all four comparisons. Eight weight-only directions reproduce the full-bond result to reported precision, as the structural rank bound predicts. Four directions outperform ordinary weight-SVD and activation-PCA bases at the same width. In the bigram-heavy intervention, the four-coordinate patch is slightly *closer to the donor's effect* than a full-bond patch, consistent with partial patching counteracting a mismatched unpatched path; it is not evidence that four dimensions represent more of the bond than all 64.

The exact observable generator supplies a separate target. Among conflict positions where both proposed tokens have positive oracle probability under both tags, the trained model's tag-shift RMSE to the oracle is **.438/.440** for induction-heavy and **.293/.301** for bigram-heavy. The four-direction weight-only *patch* has oracle-shift RMSE **.576/.584** and **.426/.434** respectively. Patching one bond therefore does **not** recover the full generative rule intervention, even though it almost saturates what this bond can mediate. On no-conflict positions whose oracle distribution is unchanged by the tag, the four-direction induction patch changes the baseline model by KL **.000614/.000682** nats, close to the full-bond patch's **.000633/.000701**; this is a local selectivity control, not proof of no collateral effects.

## Interpretation and next test

This is a concrete instance of the original [toy-language proposal](https://www.lesswrong.com/posts/GncmyuKmp526RJpmj/ambitious-mech-interp-w-tensor-transformers-on-toy-languages): a known DGP intervention supplies a causal question, and a tensor-transformer permits an exact weight-only computation of the relevant first-layer response. The exact eight-dimensional span follows from the architecture. The empirical finding is that **four weight-derived directions** suffice for nearly all measurable first-bond mediation across two tag changes and held-out conflict contexts. It does not establish unique functional canonicalization, a human-readable rule vector, or recovery of the complete model circuit.

The next discriminating test is whether a similarly compact **weight-only** basis can predict a rule change that is *not supplied as one dedicated tag token*, or recover the same composed rule effect under a larger, more natural vocabulary and a different training seed. Until then, the method is best described as intervention-specified weight-only circuit analysis, rather than task-blind discovery.

Reproduce with [the mediation script](eval_conditional_bond_mediation.py) and its four saved panels: [induction 7802](conditional_mixed_rules_v1/weight_only_tag_contrast_bond0_test7802.json), [induction 7812](conditional_mixed_rules_v1/weight_only_tag_contrast_bond0_test7812.json), [bigram 7802](conditional_mixed_rules_v1/weight_only_tag_contrast_bond0_bigram_test7802.json), and [bigram 7812](conditional_mixed_rules_v1/weight_only_tag_contrast_bond0_bigram_test7812.json). Model and generator revisions are the repository state accompanying this report.
