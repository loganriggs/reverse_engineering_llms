## H789 — Fresh lexical/format replication does not establish a transferable final-position L4 copying feature

Previous turn was progress: H788 verified native-angle endpoints and retained a small L4 copying odd-response lead on the old panel. H789 freezes the same intervention rules and evaluates fresh prompt constructions and answer vocabulary, with L1 and entity binding as comparators. No post-outcome prompts, layers or fractions were substituted.

The panel contains 16 lexical items in each of four task/format cells (64 pairs, 192 contexts). Copying uses tree/plant keys and household-object values with either vertical-bar separators or colon/newline formatting. Entity binding uses new name pairs and station/parcel or cafe/ticket wording. There is zero exact prompt overlap and zero answer-word overlap with H785. This does not establish novelty relative to every earlier experiment or model pretraining. Lexicon and formats both change relative to H785, so a failed replication does not isolate the source of shift. The two formats share lexical item indices; their samples are not independent.

The primary prediction is positive L4 copying direction-sensitive donor alignment that exceeds random controls at native-angle fractions .25/.5/1. True versus shuffled directions tests item-specificity. Full native-swap effects are reported separately. Manifest and analysis plan were saved before model evaluation. All token-valid items are used, regardless of individual correctness.

### Competence

| Task/format | Pairs | Both endpoints correct in answer contrast | Intended full-vocabulary top1 | Gate |
|---|---:|---:|---:|---|
| Copying, vertical bars | 16 | 1.00 | .906 | pass |
| Copying, colon/newline | 16 | 1.00 | 1.00 | pass |
| Entity binding, station/parcel | 16 | 1.00 | .906 | pass |
| Entity binding, cafe/ticket | 16 | 1.00 | .125 | fail |

All nuisance answer contrasts are correct; no tokenization exclusions. The failed binding format remains in the results but cannot support a successful-task interpretation.

### Primary replication outcome

L4 copying retains positive point estimates, but the pattern is weaker and not established across formats. At full native angle:

| Copy format | True odd directed mean | Shuffled odd mean | Random odd mean | Net native-swap directed mean |
|---|---:|---:|---:|---:|
| Vertical bars | +.01548 | -.00785 | -.01371 | -.01696 |
| Colon/newline | +.00848 | -.00294 | -.00145 | +.00815 |

For vertical bars, true-minus-random is +.02919 with descriptive item-bootstrap interval [.00548,.05045]. The true effect itself has interval [-.00076,.03281], and true-minus-shuffled has [-.00419,.05663]. For colon/newline, true-minus-random is +.00993 with interval [-.01967,.04156]; the true effect and true-minus-shuffled intervals also span zero.

At fractions .25/.5/1, L4 true-direction means are .00347/.00711/.01548 for vertical bars and .00218/.00433/.00848 for colon/newline. The smooth scaling of point estimates does not overcome uncertainty or establish semantic specificity. In contrast, the old H788 full-angle mean was .04576. Effects from different lexical/template panels should not be interpreted as a single controlled factor comparison.

The primary lead therefore does not clearly replicate across the predeclared formats. This is inconclusive evidence for a small general effect, not proof that the effect is exactly zero. It is insufficient to justify training a new featurizer around a supposed robust final-position L4 copying unit.

The L1 comparator has a positive signal for vertical-bar copying (odd mean .03716 and true-minus-shuffled .03473, descriptive interval [.01021,.05935]), but not colon/newline copying (odd mean .00030). This is a format-dependent exploratory comparator result, not a replacement success criterion for the primary L4 hypothesis. Its net native-swap directed mean is negative in both copy formats. We should not chase the most favorable layer/format combination.

The competent station/parcel binding format has L4 odd mean .01404, with intervals for true versus zero, shuffled and random all spanning zero. It supplies no strong cross-task confirmation. Native swap margins are often small; no ratios to small behavioral effects are used.

### Verification

192 native capture forwards plus 12,544 intervention/reference forwards = 12,736 total. Constant norm/radius errors are at floating-point precision. True fraction1 reaches the norm-scaled donor input within 3.92e-14 relative L2. Donor/receiver input norm discrepancy is <=1.60e-7. Native input and output substitutions agree within 6.68e-6 answer margin; true endpoint and actual donor-output swap agree within 7.63e-6. Identity logit discrepancy across batches <=2.91e-5. All effects finite; manifest, preregistration, runner, model and input hashes unchanged during intervention execution. Original reserved test untouched; natural-document pool remains 26,112.

Bootstrap intervals use 16 item clusters within each task/format cell, grouping reciprocal edits and random seeds. No pooled cross-format inference treats shared items as independent. Intervals are descriptive and not multiplicity-adjusted. No featurizer has been fit to this replication panel.

### Decision

Deprioritize a universal feature claim at the final-position L4 MLP. The model can copy well while local donor-direction interventions at that site are weak, format-dependent, or canceled. This does not imply that early-layer source representations are unimportant: earlier source-position interventions through the L5 value pathway probed a different route.

Next compare native donor MLP-output substitutions at the two copied-value source positions versus the final query position on this controlled copying panel. Predeclare individual-source, both-source, query and joint-source/query conditions at L1/L4; audit token alignment and exact identity controls. Read whole-model effects and keep the previously established L5 value-path evidence distinct. The purpose is to discriminate location-specific representation from a weak site choice, not to claim a manifold from another favorable average. Any newly selected site requires separate transfer confirmation before fitting or validating feature groups.
