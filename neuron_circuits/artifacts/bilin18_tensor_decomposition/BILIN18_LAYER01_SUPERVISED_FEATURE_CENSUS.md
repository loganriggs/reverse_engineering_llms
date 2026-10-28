# Supervised FineWeb feature census: layers 0–1

Fifty token-level target features were declared before activation collection. Native product-channel coalitions are selected on 16,384 discovery sites and evaluated on 16,384 disjoint held-out sites. The competence gate uses the full model's exact clipped output logits; the unigram control asks how much is predictable from the current token alone, which is especially important because this architecture reinjects the embedding at every layer.

## Locked hypotheses

- **S0_label_coverage: not supported.**
- **S1_model_competence: supported.**
- **S2_fixed_coalitions_recover_known_features: supported.**
- **S3_coalitions_beat_single_neurons: supported.**
- **S4_depth_adds_context: not supported.**
- **S5_neuron_aligned_ontology_is_sufficient: not supported.**
- **S6_embedding_reinjection_control: supported.**
- **S7_evaluation_is_held_out: supported.**

![Supervised census](bilin18_layer01_supervised_feature_census.png)

Eligible features: **32/50**. Competent features: **32/50**. Median model-output AUROC among competent labels: **0.793**; current-token unigram: **0.731**.

| Feature | Positives (D/H) | Model | Unigram | L0 one / 64 | L1 one / 64 | 90%-size L0/L1 |
|---|---:|---:|---:|---:|---:|---:|
| url_web | 3/5 | 1.000 | 0.577 | 0.966/0.952 | 0.520/0.852 | 1/16 |
| percent | 8/8 | 0.999 | 0.664 | 0.934/0.994 | 0.961/0.992 | 4/1 |
| question | 27/40 | 0.991 | 0.650 | 0.487/0.699 | 0.549/0.737 | 64/64 |
| closing_delimiter | 40/36 | 0.988 | 0.706 | 0.615/0.842 | 0.651/0.795 | 64/256 |
| exclamation | 23/21 | 0.985 | 0.659 | 0.532/0.704 | 0.538/0.753 | 256/4 |
| weekday | 27/15 | 0.955 | 0.789 | 0.560/0.630 | 0.706/0.650 | 256/1 |
| word_continuation | 1837/1864 | 0.954 | 0.896 | 0.696/0.850 | 0.648/0.924 | 64/64 |
| modal | 150/169 | 0.942 | 0.707 | 0.663/0.747 | 0.710/0.723 | 64/256 |
| comma | 652/603 | 0.928 | 0.741 | 0.701/0.765 | 0.690/0.793 | 16/16 |
| colon_semicolon | 82/93 | 0.927 | 0.720 | 0.508/0.783 | 0.641/0.796 | 256/64 |
| all_caps | 163/320 | 0.920 | 0.683 | 0.603/0.774 | 0.712/0.801 | 16/64 |
| numeric | 310/436 | 0.919 | 0.726 | 0.640/0.806 | 0.634/0.835 | 64/64 |
| period | 614/630 | 0.916 | 0.749 | 0.655/0.772 | 0.676/0.788 | 64/64 |
| month | 35/58 | 0.912 | 0.655 | 0.655/0.779 | 0.641/0.695 | 256/256 |
| time_unit | 81/91 | 0.903 | 0.688 | 0.754/0.825 | 0.671/0.820 | 64/64 |
| slash | 35/32 | 0.897 | 0.716 | 0.588/0.788 | 0.664/0.768 | 256/256 |
| capitalized | 2212/2349 | 0.885 | 0.763 | 0.681/0.693 | 0.630/0.789 | 256/64 |
| suffix_ment | 57/35 | 0.884 | 0.603 | 0.600/0.661 | 0.531/0.687 | 4/64 |
| currency | 7/21 | 0.883 | 0.536 | 0.412/0.383 | 0.375/0.438 | 4/256 |
| suffix_ity | 47/43 | 0.882 | 0.612 | 0.541/0.639 | 0.484/0.666 | 256/256 |
| newline | 387/329 | 0.879 | 0.871 | 0.807/0.866 | 0.768/0.852 | 16/4 |
| opening_delimiter | 65/64 | 0.869 | 0.665 | 0.534/0.703 | 0.528/0.754 | 256/256 |
| quote | 51/41 | 0.865 | 0.632 | 0.483/0.633 | 0.537/0.553 | 256/256 |
| negation | 74/80 | 0.845 | 0.699 | 0.647/0.849 | 0.670/0.832 | 64/256 |
| suffix_ed | 415/343 | 0.843 | 0.708 | 0.589/0.712 | 0.625/0.696 | 64/64 |
| lowercase | 10986/10878 | 0.817 | 0.749 | 0.608/0.693 | 0.603/0.739 | 256/256 |
| code_operator | 244/298 | 0.814 | 0.752 | 0.634/0.786 | 0.541/0.796 | 64/64 |
| suffix_ion | 171/171 | 0.801 | 0.612 | 0.534/0.607 | 0.550/0.625 | 256/256 |
| suffix_ous | 20/11 | 0.797 | 0.514 | 0.662/0.704 | 0.587/0.748 | 16/64 |
| non_ascii | 496/353 | 0.785 | 0.832 | 0.724/0.780 | 0.720/0.771 | 16/64 |
| suffix_ive | 28/29 | 0.778 | 0.428 | 0.618/0.747 | 0.571/0.671 | 64/256 |
| quantity_unit | 55/47 | 0.770 | 0.546 | 0.522/0.603 | 0.614/0.709 | 256/256 |
| conjunction | 574/578 | 0.770 | 0.690 | 0.542/0.626 | 0.567/0.700 | 256/256 |
| suffix_ing | 375/348 | 0.769 | 0.646 | 0.598/0.608 | 0.546/0.646 | 16/64 |
| auxiliary | 737/743 | 0.768 | 0.754 | 0.626/0.732 | 0.595/0.738 | 64/256 |
| leading_space | 11742/11813 | 0.764 | 0.828 | 0.736/0.828 | 0.585/0.829 | 16/64 |
| preposition | 1589/1586 | 0.763 | 0.736 | 0.680/0.752 | 0.668/0.770 | 16/64 |
| suffix_s | 1046/1018 | 0.763 | 0.631 | 0.525/0.664 | 0.553/0.666 | 64/64 |
| pronoun | 667/727 | 0.758 | 0.748 | 0.644/0.749 | 0.564/0.764 | 16/256 |
| suffix_ness | 21/17 | 0.756 | 0.518 | 0.616/0.611 | 0.438/0.751 | 1/64 |
| alphabetic | 13200/13233 | 0.742 | 0.736 | 0.677/0.725 | 0.559/0.656 | 64/256 |
| suffix_ly | 187/158 | 0.738 | 0.583 | 0.577/0.664 | 0.578/0.691 | 64/64 |
| punctuation | 2361/2262 | 0.734 | 0.755 | 0.703/0.737 | 0.588/0.636 | 64/256 |
| dash | 207/184 | 0.728 | 0.750 | 0.620/0.747 | 0.531/0.787 | 256/256 |
| determiner | 1513/1472 | 0.706 | 0.763 | 0.692/0.752 | 0.596/0.807 | 64/64 |
| boolean | 148/148 | 0.684 | 0.633 | 0.578/0.703 | 0.603/0.709 | 64/256 |
| long_piece | 1528/1487 | 0.668 | 0.639 | 0.588/0.624 | 0.558/0.669 | 256/64 |
| programming_keyword | 341/388 | 0.654 | 0.582 | 0.612/0.654 | 0.563/0.650 | 64/256 |
| tab | 0/0 | — | — | —/— | —/— | None/None |
| alphanumeric | 0/0 | — | — | —/— | —/— | None/None |

## Interpretation guardrails

- These labels are supervised probes of a declared ontology, not proof that the model internally uses those exact concepts.
- A coalition is a fixed linear discriminator over native product channels. It is not yet the input-conditioned causal coalition tested by edit compilation.
- Current-token unigram prediction is a strong architectural control, but it does not exhaust n-gram or embedding-based baselines.
- The output-logit competence score tests whether the trained model ranks the feature; it is not an exact class-probability calibration metric.
