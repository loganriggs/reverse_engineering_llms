## H791 — Fresh source swaps carry donor-specific content, alongside broad alternative promotion

Previous turn was progress: H790 localized stronger effects to L1 value-source positions, but answer-margin changes could not separate actual donor promotion from receiver suppression or repeated-value inconsistency. H791 fixes the source sites and tests a fresh lexical panel with candidate-resolved readouts.

The new prompts use Greek-label keys and16 transport-value words, keeping the two prior formats (vertical bars; colon/newline) fixed. There are16 items per format and four native variants per item: one receiver word, one designated alternate, and two additional donor words. Donor variants are fixed offsets3/7/11 in the value list; each is a legitimate value for the same key. They test whether effects follow the actual donor rather than any arbitrary alternate answer. We replace native L1 MLP outputs at the first source, second source, both sources, or final query token. We record logits and full-vocabulary-normalized log-probabilities for all16 predetermined candidates, not just a binary margin.

All128 native contexts and512 interventions were run. There are no token/alignment exclusions, no exact prompt overlap with H789, and no candidate-answer-word overlap with H789. Formats share lexical item indices and candidate words recur across items. This is fresh lexical confirmation, not an independent test of new formatting or other task families.

Both formats pass competence: all four native variants choose the correct candidate for every item; pooled intended full-vocabulary top1 is96.875% for vertical bars and100% for colon/newline. No outcome-based prompt or candidate selection was performed.

### Content-specificity result

Source interventions promote the actual donor word beyond a general loss of receiver confidence. Mean log-probability changes (natural-log units) and donor specificity are:

| Format | Site | Receiver log-probability change | Donor log-probability change | Donor-vs-receiver margin change | Donor-specific contrast |
|---|---|---:|---:|---:|---:|
| Vertical bars | First source | +.056 | +.773 | +.717 | +.333 |
| Vertical bars | Second source | -.016 | +.546 | +.561 | +.201 |
| Vertical bars | Both sources | -.096 | +.868 | +.964 | +.510 |
| Vertical bars | Query | +.011 | -.003 | -.014 | +.003 |
| Colon/newline | First source | -.123 | +.759 | +.882 | +.284 |
| Colon/newline | Second source | -.137 | +.477 | +.614 | +.165 |
| Colon/newline | Both sources | -.066 | +.666 | +.732 | +.417 |
| Colon/newline | Query | +.009 | +.008 | -.001 | +.004 |

The donor-specific contrast is the donor's logit change minus the mean logit change of the14 candidates other than donor and receiver. The common log-probability normalization cancels in this contrast, so it is also the corresponding log-probability contrast. It tests specificity against this candidate set, not against every vocabulary item.

For both-source swaps, descriptive item-bootstrap intervals for specificity are [.416,.599] in vertical bars and [.320,.505] in colon/newline. All six source/site-format specificity intervals are positive; query effects are tiny. Thus pure receiver suppression is insufficient to explain the source results. Nevertheless, broad alternative promotion also occurs: the donor's log-probability increase is larger than its specificity advantage, and unrelated alternatives can improve substantially.

### Do effects follow the actual donor?

When donor variants2/3 are substituted, compare their actual word with the designated but unswapped alternate (variant1). Both-source swaps favor the actual donor by +.410 in vertical bars (interval [.229,.593]) and +.274 in colon/newline (interval [.078,.472]). The fixed alternate also gains relative to the receiver (+.536 and+.425 respectively), demonstrating a mixture of actual-donor specificity and broad alternative effects.

Single-source discrimination against that fixed alternate is weaker and not uniform: three of four single-source/site-format intervals span zero. The actual donor is the candidate with the largest nonreceiver logit increase in only33.3%/35.4% of joint-source interventions. These data do not support a perfectly selective dictionary lookup or clean content replacement.

### Behavioral limits and the prior joint-source hypothesis

No intervention changes the highest-scoring candidate away from the receiver or makes the donor the full-vocabulary top token. With both-source swaps, mean donor probability rises from .000571 to .000914 in vertical bars and from .000480 to .000633 in colon/newline. Mean receiver probability changes from .436 to .418 and .498 to .484. Donor promotion is measurable but small in absolute probability; this is not transplantation of the complete copying computation.

The earlier finding that both-source swaps were weaker than either single-source swap does not reproduce uniformly. On this panel, joint-source donor-margin gain exceeds both single-source gains in vertical bars and lies between them in colon/newline. Joint-source donor specificity is stronger than either single-source specificity in both formats. This weakens a universal “single-source inconsistency explains the entire effect” account. Context-dependent interactions and broad alternative promotion remain possible. The present panel uses three donors for each designated receiver rather than the earlier reciprocal-pair design, so it is not a controlled decomposition of which distribution change caused that difference.

### Audits

640 total model forwards. Requested output replacements are exact. Identity logits replay within1.77e-5 across batches. The independently computed logit-margin change and difference of log-probability changes agree within1.64e-6. All effects finite; model, manifest and runner hashes unchanged. Original reserved test untouched; natural-document pool remains26,112.

Bootstrap intervals use16 item clusters per format and average donor conditions within item. Shared vocabulary and shared cross-format items limit independence; intervals are descriptive without multiplicity correction. Native patch norms are saved and differ across sites/donors, so no matched-energy causal-efficiency claim follows. All16 candidate responses are saved for further analysis.

### Research decision

There is now a better-supported causal target for representation research: L1 source-output changes contain transferable information about which value was substituted, although the effect is partial and accompanied by nonspecific changes. This is evidence of content-bearing representations at a location, not yet evidence for a low-dimensional manifold, meaningful group partition, or computational DAG.

Next cross the donor identities at the two source positions: coherent A/A and B/B versus mixed A/B and B/A, plus single-source and receiver controls. Track promotion of each actual donor, unrelated candidates and the receiver. This will distinguish independent content contributions from an agreement-sensitive computation and provide a concrete compositional test for a shared dictionary or DAG. Preserve the native-output reference before asking a compressed featurizer to reproduce the identity-specific part of the response.
