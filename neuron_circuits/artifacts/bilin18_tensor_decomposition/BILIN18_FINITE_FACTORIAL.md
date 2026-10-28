## H658 — Exact incoming state plus attention restores finite task outputs

H657 showed that finite later-MLP context edits only partially reproduce natural task-prompt distributions. H658 localizes the remaining mismatch with all eight combinations of three exact donor replacements, on the same96 held-out copying prompts and paired second-wording task prompts:

- I: replace the final-position residual leaving block9 with the task prompt's residual.
- A: replace the final-position attention output at each block10–17 with the task prompt's attention output.
- C: replace the final-position normalized MLP input at each block10–17 with the task prompt's input.

All other computation runs normally. Inputs and final answer suffixes match within pairs. Final-position initial embedding residual x0 is asserted identical, including across different prompt lengths. Donor attention outputs are captured from fresh native task forwards; saved donor residuals and MLP inputs are reused. Every combination, family, and item is retained. These are donor-informed localization controls, not fitted predictors or deployable compact representations.

**KL(task||edited), nats; lower means closer to the natural task output:**

| Family | Copy | I | A | C | I+C | A+C | I+A | I+A+C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|addition|0.1415|0.0419|0.0488|0.0386|0.0192|0.0115|0.0000|0.0000|
|agreement|0.1837|0.0875|0.0373|0.0748|0.0594|0.0079|0.0000|0.0000|
|antonym|0.1381|0.0758|0.0401|0.0777|0.0482|0.0189|0.0000|0.0000|
|capital|0.2041|0.0962|0.0910|0.0888|0.0525|0.0177|0.0000|0.0000|
|plural|0.1685|0.0817|0.0350|0.0665|0.0419|0.0104|0.0000|0.0000|
|successor|0.5062|0.1366|0.1540|0.1586|0.0574|0.0570|0.0000|0.0000|

The complete replacement recovers donor output to maximum KL8.77e-12 and maximum logit error1.39e-5. I+A also recovers donor output to maximum KL7.43e-12, without direct MLP replacement. This is expected from the architecture, not discovery of a minimal two-unit circuit: matching the incoming final-position state, identical x0 and each attention write makes the MLP inputs match, so deterministic MLP computation follows naturally. It validates the intervention boundary and confirms that direct MLP replacement is redundant when its causal inputs are restored. The transplanted attention writes still carry substantial donor computation and are not compact or independent units.

Attention alone reduces mismatch in all families and outperforms MLP-context-only replacement in five of six (addition is the exception). A+C leaves less KL than I+C in five families; successor is almost tied. These conditional comparisons identify attention as an important missing target for the previous MLP-only approximation, but do not assign causal percentages or establish a unique decomposition. Residual differences and nonlinear interactions remain.

**Loss and fidelity remain distinct.** Examples: for antonyms, I decreases output KL from0.138 to0.076 while worsening target CE from5.387 to5.880. A decreases KL to0.040 and improves target CE to5.094, whereas complete recovery gives the natural task's worse CE5.593. For capitals, A+C has target CE4.967, slightly better than the fully recovered task's4.986 despite its remaining KL0.0177. Exact recovery is not equivalent to optimizing declared-answer loss. All conditions remain at0% exact next-token accuracy on this held-out subset; natural task outputs are not necessarily correct.

The current result changes the approximation target. Rather than continuing to refine a separately transplanted MLP-context shift, test a data-based description of the block9 boundary change and later attention-output changes, letting MLP computation run naturally. Fit original-wording discovery means, evaluate second-wording held-out items, and cross predicted versus exact boundary/attention replacements to see which component limits transfer. Include other-task means and native baselines. Compare finite output KL, target CE and correctness independently. If simple predictions work, this gives a more causally coherent conditional surrogate; if they fail, exact recovery still supplies a useful upper reference without pretending the donor-informed states are an explanation.

This does not prove a global canonical form, smallest feature unit, semantic primitive, or end-to-end model compression. The exact information transplanted here is rich and input-specific; a useful representation must predict or compress it on independent data. The user’s broader objective remains open.

Audits:864 whole-model forwards (96 donor capture plus8 conditions x96 receivers). Native-copy and native-task replay deviations below1.86e-5 and1.53e-5 respectively; full-recovery logit deviation1.39e-5. Prior MLP-only KL reproduces within1.75e-6. Final x0 equality asserted; all runtime hashes unchanged; all metrics finite. No cases removed. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
