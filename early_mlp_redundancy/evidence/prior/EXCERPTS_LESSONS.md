# Selected historical excerpts (not a new endorsement)


## LESSONS.md, original lines 578–608

578: I measured mlp0's token-table ceiling with a table that covered 76.6% of eval positions.
579: The uncovered positions got a fallback value. I then excluded those positions from the CE
580: average, believing that isolated the coverage problem.
581: 
582: It does not. The substituted forward pass still ran with WRONG mlp0 outputs at the
583: uncovered positions, and attention in layers 1-17 mixed that error into the predictions
584: at the covered positions. The score-time mask removes the directly-damaged rows and
585: leaves all of the indirect damage.
586: 
587: - **How wrong it was:** the frozen-attn0 arm has a KNOWN ANSWER of 1.0. Substituting
588:   everywhere and masking the score gave 55.83%. Substituting only where covered gave
589:   100.00% (ce_table == ce_live to five decimals). A 44-point error, in a quantity whose
590:   true value was derivable in advance.
591: - **Why it fooled me twice:** the first version's headline moved 25 points in the
592:   DIRECTION OPPOSITE the hypothesis, which reads as a strong negative result rather than
593:   as a bug. The second version's "obvious fix" moved it 6 points and still read as a
594:   clean negative. Neither looked like an artifact from the outside.
595: - **What caught it:** an instrument check with an answer known before the run. Not a
596:   sanity check on the output's plausibility -- a quantity I could derive analytically
597:   (attn0 constant + position-wise MLP => mlp0 is a token function => a covered table is
598:   exact) and refuse to proceed past. It is the only reason the 25-point "finding" is not
599:   in FINDINGS.
600: - **The rule:** when substituting position-wise, FIX THE FORWARD PASS, do not repair the
601:   score. Apply the intervention only where it is valid and leave the module live
602:   elsewhere (`torch.where(valid, substitute, out)`). In a model with attention, no
603:   post-hoc row mask can undo a bad value that has already been mixed forward.
604: - **Scope:** this contaminates every table-ceiling in this project that used an
605:   unseen-token fallback and substituted at all positions -- they are UNDERSTATED. On
606:   mlp0 the understatement was 15.9 points (74.42% -> 90.27%).
607: 
608: ## 28. A SUBSTITUTION FITTED AGAINST THE REAL MODEL AND APPLIED JOINTLY COMPOUNDS OFF-DISTRIBUTION — and past some number of sites it goes negative (§1668)

## LESSONS.md, original lines 1455–1472

1455: ## LESSONS 57 — a share-of-the-total bar cannot separate DOMINANCE from REDUNDANCY
1456: 
1457: §1830 asked which site of layer 1 buys the 38.9pp drop, and registered pred_a as: *the larger of the two
1458: single-site drops is at least 70% of the joint drop.* It passed at **99.5%**. It should not have been
1459: counted as evidence for anything.
1460: 
1461: The result was that **both** sites individually cost ~100% of the joint drop — attn1 +37.4pp, mlp1
1462: +38.7pp, joint +38.9pp. That is redundancy, and it is the opposite of the "one site dominates" the bar
1463: was written to detect. **A threshold on `max(a, b) / joint` is satisfied identically by "a is everything
1464: and b is nothing" and by "a and b are each everything".** The statistic has no power to distinguish them
1465: because it never looks at the smaller value.
1466: 
1467: The prediction that carried the finding was pred_c, which asked about **additivity**: `(a + b) / joint`,
1468: which came out **1.96x** against 2.00x for exact redundancy. That single ratio separates all three
1469: outcomes — ~1.0x additive, ~2.0x redundant, >2.0x super-additive — where the share bar separates none.
1470: 
1471: This is a sharper instance of LESSONS 44 (a binary predicate cannot separate three outcomes) and of
1472: LESSONS 48 (a marginal along one ordering is not an attribution to a component): the failure is not that
