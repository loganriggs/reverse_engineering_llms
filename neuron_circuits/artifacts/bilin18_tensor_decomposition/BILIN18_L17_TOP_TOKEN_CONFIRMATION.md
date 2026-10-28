

## H384 — Fresh-document confirmation of the fixed top512 approximation

Freeze K=512 from H383 and compare against full-recipient probability fitting. Use 512 fresh complete documents, excluding all 5,312 complete-document hashes from the listed prior panels. Four positions per document; heads 0/1/6; actual responses, six newly seeded moment draws, six fresh cyclic donor assignments. Retain both frozen probability/random bases and ranks 16/64. Both methods omit the cap derivative during fitting and receive identical full-vocabulary native finite KL scoring and the eight-step safeguard. Top512 probabilities are selected from the intact recipient distribution and renormalized. The full native response is still needed for fitting and step selection.

The preregistered scientific gate requires all 36 full-minus-top512 simultaneous upper bounds to be below five percentage points; rank correctness is a separate audit. Result: {'fresh_top512_within5': True, 'rank_bound': True}.

|Basis|Head|Family|Full16|Top51216|Full64|Top51264|
|---|---|---|---:|---:|---:|---:|
|probability|0|actual|0.7292|0.7225|0.8599|0.8480|
|probability|0|moment|0.5587|0.5531|0.7478|0.7330|
|probability|0|donor|0.5598|0.5541|0.7479|0.7332|
|probability|1|actual|0.6447|0.6346|0.8012|0.7826|
|probability|1|moment|0.6144|0.6054|0.7901|0.7722|
|probability|1|donor|0.6161|0.6065|0.7919|0.7735|
|probability|6|actual|0.6635|0.6537|0.8158|0.7977|
|probability|6|moment|0.6238|0.6203|0.8352|0.8209|
|probability|6|donor|0.6248|0.6211|0.8396|0.8247|
|random|0|actual|0.6039|0.5972|0.8132|0.8015|
|random|0|moment|0.5173|0.5127|0.7287|0.7179|
|random|0|donor|0.5151|0.5104|0.7272|0.7162|
|random|1|actual|0.5917|0.5847|0.7737|0.7609|
|random|1|moment|0.5732|0.5674|0.7610|0.7489|
|random|1|donor|0.5762|0.5701|0.7638|0.7516|
|random|6|actual|0.5483|0.5390|0.7738|0.7567|
|random|6|moment|0.5514|0.5518|0.8056|0.8001|
|random|6|donor|0.5647|0.5651|0.8128|0.8077|

Across all settings, point loss is -0.043–1.861 percentage points; the largest simultaneous upper loss is 2.294 points. Full rank64 recovery spans 72.72–85.99%; top512 spans 71.62–84.80%. Mean retained intact probability mass is 95.34%.

Uncertainty: 4,000 paired recipient-document/control-draw bootstrap resamples, with a simultaneous max-absolute-deviation band over all 36 prespecified contrasts. Bases and donor assignments remain fixed. This is confirmation on fresh documents within the same cached corpus, not a model-training holdout, new model, domain transfer test, or per-site guarantee. H383’s failed 144-contrast gate remains a failure; this is a separately frozen fresh experiment.

Audits: complete-document and sampled-window hashes independently reconstructed; all 512 new document hashes unique and disjoint from the 5,312 excluded hashes. All 16/64 Gram directions retained for both metrics. Normal-equation relative residuals [1.3172023800325693e-12, 2.5176372849344864e-13]; payload norm/angle error 4.96e-07. Saved finite grid-zero scores agree across methods and bases; minima exactly reproduce saved guarded errors. All scores finite, local fits no worse than Euclidean coefficients in their fitting metric. Per-site ranks, norms, probability mass, grid scores, chosen steps, response arrays and bootstrap samples are retained. Independent audit fields were added by work/audit_h384.py after the runner.

Interpretation: the fixed top512 approximation passes fresh-document aggregate noninferiority at the prespecified five-point margin. The worst point loss is 1.861 percentage points and the worst simultaneous upper bound is 2.294 points. One setting has a small point improvement (0.043 points), so this does not imply the full local metric always produces the best finite safeguarded fit. The fresh result supports the small aggregate loss seen in H383 without changing its failed discovery gate.

Top512 retains 95.34% of intact probability mass on average. Rank64 native recovery is 71.62–84.80% with top512 versus 72.72–85.99% with the full metric. Combined with H382 and H383, this favors context-specific token-output constraints, numbering in the hundreds in this approximation, over either concentration alone or a handful of high-probability tokens. It does not establish that 512 tokens are minimal, that each token is an independent constraint, or that these are semantic units. The same pattern in random spans remains a warning against interpreting oracle reconstruction as feature discovery.

Twenty updated hypotheses and discriminators (qualitative priorities, not posterior probabilities):

|#|Hypothesis|Current evidence / next discriminator|
|---:|---|---|
|1|Hundreds of context-selected output constraints explain much of oracle recovery.|Supported by fresh top512 confirmation; test transfer across contexts/models.|
|2|A handful of likely tokens suffice.|Disfavored under the current solver by top8/top32 losses; test whether explicit regularization changes this.|
|3|Output concentration alone determines recovery.|Disfavored by H382 exact-spectrum permutation; preserve concentration in further controls.|
|4|Token identity alignment matters chiefly among the top32.|Unresolved; preserve top32 and permute only lower-probability identities.|
|5|Tail identities carry additional target-specific directions.|Plausible; selective tail reassignment should damage recovery even with top32 unchanged.|
|6|Tail covariance mainly regularizes weakly constrained coefficients.|Plausible; replace tail covariance with a trace-matched shared or isotropic metric.|
|7|The tail matters primarily through its contribution to the target cross-moment.|Unresolved; independently swap the tail Gram matrix and target right-hand side.|
|8|Top-versus-tail mean differences matter more than within-tail covariance.|Unresolved; exact conditional-covariance decomposition separates these terms.|
|9|Required token count follows retained mass rather than a fixed K.|Plausible; freeze mass thresholds and compare costs across entropy strata.|
|10|Required token count follows effective covariance rank rather than mass.|Plausible; compare spectral effective rank with approximation loss on untouched data.|
|11|Frozen probability axes identify privileged computational units.|Weakened by high random-span oracle recovery; test target-free coefficients.|
|12|Many bases work because likely-token constraints admit flexible solutions.|Supported as a geometric alternative; characterize solution freedom and unseen-token effects.|
|13|Minimum-norm selection causes much of the small-K deficit.|Plausible; compare prespecified regularizers and null-space choices.|
|14|Rare-token finite excursions explain some approximation losses.|Supported by earlier tail diagnostics; stratify guarded/unguarded failures without exclusions.|
|15|The finite step safeguard supplies much of the apparent simplicity.|Unresolved quantitatively here; evaluate a fixed target-free step rule on fresh data.|
|16|A shared coefficient map can replace recipient adaptation.|Disfavored by H381; test richer input-conditioned maps rather than a single shared map.|
|17|Recipient features can predict oracle coefficients without observing the response.|Central unresolved computational hypothesis; train on separate documents and test native KL.|
|18|Head-specific mechanisms require different constraint sets.|Possible despite pooled success; compare preregistered head-specific residual errors.|
|19|The result depends on this final-layer output geometry rather than a general unit of computation.|Not resolved; repeat matched diagnostics at earlier layers with downstream propagation.|
|20|Conditional covariance plus a response generator is a more useful description than fixed semantic axes.|Compatible with evidence, not established; compare executable predictors at matched complexity.|

Highest-information next experiment: preserve the top32 token identities/probabilities and intervene only on the remaining probability-to-token assignments. Pair that with a decomposition of the fitting normal equations into leading-token, tail covariance, and target-cross-moment terms. This distinguishes specific tail information from stabilization; arbitrary full-spectrum shuffling cannot isolate that distinction. Freeze controls and thresholds before looking at outcomes. Then prioritize a genuinely target-free coefficient/step predictor: the broad computational-unit question remains unresolved while recovery requires the full native response.

The current fresh panel is now used evidence. The next fresh panel must exclude all 5,824 complete-document hashes, including this one. Broader goal remains active.
