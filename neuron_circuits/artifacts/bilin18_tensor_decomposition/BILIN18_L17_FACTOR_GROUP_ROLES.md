

## H412 — Fixed factor groups have source-dependent response roles

Reuse H411512 documents and frozen H410 sets: core64, added64, union128. No new selection or documents; this is a diagnostic continuation. For each head source at unit amplitude, decompose the local centered native-logit derivative into core c, additions a and remainder o. In the native Fisher metric, full energy equals C+A+O+2<c,a>+2<c,o>+2<a,o>. The derivatives include final RMS, native unembedding and logit cap. The cross terms express alignment in this output metric, not a newly identified nonlinear interaction between factors; the bilinear-factor output sum is additive before normalization.

Compute additions-only finite native prediction KL and baseline-preserving clamp KL. Pair these with the exact saved H411 core-only and union results, verifying native effect replay. A local derivative cross term need not equal the finite KL improvement. Pooled cross/full ratios and recovery fractions retain their stated denominators; none is a unique additive mediation decomposition.

Predeclare one joint27 max-absolute95% band from4000 paired document bootstrap draws: nine core/additions cross/full ratios, nine additions-only finite recoveries, and nine additions−core finite recoveries. Gates require head6 cross upper<−.05; both heads3/4 additions−core recovery lower>.1; all nine additions-only recovery lower>.8; and at least one source cross lower>.05. The last gate tests an appreciable positive alignment somewhere, not universal sign reversal. Numerical closure<1e-8 and H411 native replay<1e-10 are separate checks.

Gates: {'energy_identity': True, 'native_replay': True, 'head6_cancellation05': False, 'heads3_4_additions_over_core10': False, 'all_additions80': False, 'any_positive_cross05': False}.

|Head|Core-add cross/full|Joint27 interval|Added64 finite recovery|Joint27 interval|Added−core recovery interval|
|---|---:|---|---:|---|---|
|0|0.050483|[-0.22207946764828404, 0.32304463082351637]|0.150200|[-0.12236187188868175, 0.4227622265831187]|[-0.5090670847472484, 0.036057013724552056]|
|1|0.088870|[-0.18369176652673455, 0.36143233194506585]|0.259329|[-0.01323271267691084, 0.5318913857948896]|[-0.5083233333955278, 0.03680076507627261]|
|2|0.035420|[-0.23714165138290255, 0.3079824470888979]|0.309072|[0.03651026589815076, 0.5816343643699512]|[-0.2399098539749741, 0.3052142444968263]|
|3|0.031854|[-0.24070801154158952, 0.30441608693021094]|0.391779|[0.11921677757843485, 0.6643408760502353]|[-0.19053930938497898, 0.35458478908682145]|
|4|-0.011358|[-0.2839199810421179, 0.2612041174296825]|0.494248|[0.22168602138277493, 0.7668101198545754]|[0.2178767619480575, 0.7630008604198579]|
|5|0.048148|[-0.2244142817868942, 0.3207098166849062]|0.147835|[-0.12472730028355461, 0.4203967981882458]|[-0.43041914952460225, 0.11470494894719818]|
|6|-0.228986|[-0.5015478755816899, 0.04357622289011051]|-0.113674|[-0.38623575318883896, 0.15888834528296147]|[-1.0376429422869746, -0.4925188438151743]|
|7|0.126741|[-0.14582095472224868, 0.3993031437495518]|0.232066|[-0.04049645084747344, 0.504627647624327]|[-0.5429481987004366, 0.0021758997713637873]|
|8|0.121041|[-0.1515211742672215, 0.39360292420457893]|0.178202|[-0.09436004060063719, 0.45076405787116325]|[-0.8808726868747782, -0.3357485884029777]|

|Head|Core64 recovery|Added64 recovery|Union128 recovery|Added64 clamp remaining|Pooled core/add Fisher cosine|
|---|---:|---:|---:|---:|---:|
|0|0.386705|0.150200|0.489253|0.848340|0.379287|
|1|0.495091|0.259329|0.677307|0.737484|0.292675|
|2|0.276420|0.309072|0.546523|0.688894|0.109466|
|3|0.309756|0.391779|0.665927|0.605843|0.110208|
|4|0.003809|0.494248|0.508205|0.513249|-0.087598|
|5|0.305692|0.147835|0.406987|0.851128|0.270326|
|6|0.651407|-0.113674|0.811105|1.102337|-0.336478|
|7|0.502452|0.232066|0.607940|0.767486|0.443143|
|8|0.786513|0.178202|0.842506|0.819817|0.600139|

The pooled cosine is cross/(2sqrt(C A)), using pooled energies; it is not a mean of site cosines. Clamp and cosine summaries are descriptive. Audits: {'Fisher_energy_identity_max_relative': 1.0611690504002034e-14, 'direct_additions_clamp_state_max_absolute': 5.0067901611328125e-06, 'finite_core_plus_add_union_relative_error': 7.437508884322597e-07, 'native_effect_replay_max_absolute': 0.0}. Independent saved-array checks: {'mean_metrics_replay': True, 'three_part_energy_max_relative_error': 9.219993388723078e-15, 'native_effect_bitwise_H411_replay': True, 'primary_stats_replay': True}.

Outcome: numerical closure and native replay pass; all four scientific gates fail. The single joint27 band has radius0.272562, so neither the head6 cancellation threshold nor an appreciable positive cross term somewhere is established at the prespecified simultaneous confidence level. Do not replace this band after seeing the outcomes.

Nevertheless, the measured role pattern is informative. Head6 core/additions cross/full is−0.228986,[−0.501548,0.043576]; head8 is+0.121041,[−0.151521,0.393603]. Their pooled Fisher cosines are−0.336478 and+0.600139 respectively. These are suggestive opposite alignments, not confirmed cross-term signs under the joint27 band. At head6, additions-only finite recovery is−0.113674,[−0.386236,0.158888], versus core64 0.651407 and union128 0.811105. Negative recovery means this additions-only prediction is worse than the intact/no-edit prediction under the stated target KL; it does not mean a negative causal effect. At head8, corresponding recoveries are0.178202,0.786513 and0.842506.

Head4 shows stronger standalone evidence for the added group: additions-only recovery0.494248,[0.221686,0.766810], versus core0.003809 and union0.508205. Its additions-minus-core difference is0.490439,[0.217877,0.763001], exceeding10 points individually within the predeclared joint27 family. Head3's difference is only0.082023,[−0.190539,0.354585]; therefore the combined heads3-and4 gate fails. Head6 and head8 additions-minus-core intervals are entirely negative: respectively[−1.037643,−0.492519] and[−0.880873,−0.335749]. Thus the same fixed64 added factors predict more of head4's response than the core, while predicting less of head6/head8 than the core. This supports source-dependent usefulness without proving the proposed cancellation-sign contrast.

Uncertainty audit, descriptive and with no exclusions: the head3 additions-minus-core statistic sets the maximum bootstrap deviation in2948/4000 resamples; head2 does so in559, head5 in219. One document contributes15.327% of the pooled head3 native effect, compared with2.372% for head6 and1.269% for head8. This explains why a single unstandardized simultaneous band is broad even for other comparisons. It does not justify removing the document, changing the band or reporting a narrower post-hoc interval as confirmation. All rows remain included.

The three-part energy sum closes within1.06e-14 relative error. Native finite effects replay H411 bitwise. Core plus added finite raw vectors agree with union128 within7.44e-7 relative error, and direct additions-clamping state calculations agree within5.01e-6 in the checked sites. These numerical audits validate accounting; they do not turn the output-metric cross terms into semantic interactions or establish a unique computational unit.

Hypothesis update: #5 (compensating groups) remains supported by earlier rescue, with a more specific source-dependent alignment hypothesis now requiring confirmation. #11/#14 (source scope/shared transfer) sharpen: the fixed added group has very different standalone value across sources. #9/#13 (denominator/context effects) deserve attention for head3, because a few contexts can dominate a source-level ratio. #19 (semantic features) remains untested. A single scalar importance or one universal label for the added group would conceal these different response roles.

Next: freeze three selected contrasts on a new512-document panel excluding all8384 previous documents: head6 core/add cross/full upper<−.05, head8 cross/full lower>.05, and head4 additions-minus-core finite recovery lower>.1. Use one predeclared joint3 band and preserve the selection history: these are H412-selected hypotheses, to be confirmed on new data rather than reinterpreting this panel. Also retain unfiltered document-level effect and error contributions to audit concentration. Head3 remains unresolved; do not silently include it in the successful standalone-role claim. If the three contrasts confirm, test whether alignment follows source identity or recipient context through controlled swaps, before undertaking another broad factor-selection search.

No new documents were consumed in H412. The next fresh panel must exclude8384 completed-panel documents. The broad goal remains open, and no experiment processes remain running.
