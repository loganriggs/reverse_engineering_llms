## H980 — Fresh panel has zero prompt overlap and six native-passing cohorts

Previous turn was progress: H979 produced a single-state quadratic readout on the development panel. Its model artifact hash was fixed before this panel. Twelve adjective pairs new to checked H962/H975 panels, in three fixed cohorts, crossed with film/book/story/I-thought templates and four adjective-polarity/negation states:48 quartets,192 unique prompts. Exact prompt overlap with checked H962/H966/H975 panels is zero. Novelty is relative to research panels, not all historical work or model training.

Fresh1: absorbing/tiresome, captivating/lifeless, inspiring/depressing, imaginative/formulaic. Fresh2: refreshing/stale, polished/sloppy, stunning/hideous, charming/annoying. Fresh3: uplifting/miserable, exquisite/repulsive, moving/forgettable, enchanting/obnoxious. No post-outcome replacements.

Gate unchanged: >=4 quartets, >=75% allfour binary correct, >=50% intended full-vocabulary top1. Allthree book cohorts and allthree story cohorts pass. Film and I-thought cohorts fail full-vocabulary threshold despite mostly high binary correctness. Book/story fresh2 are clean single-token cohorts: allfour quartets binary-perfect and75% full-vocabulary top1. Fresh1 has onlyone single-token quartet; fresh3 none. Locked last-subtoken alignment is retained.

Audits:193/193 forwards, repeat logit replay1.72e-5, no token-boundary exclusions, zero prior prompt overlap, unchanged input/manifest hashes and frozen model hash recorded before evaluation. No fitting, intervention, or reserved-test access. Natural-document pool26,112.

Next: apply the exact H979 model artifact unchanged to all48 quartets and evaluate scalar-output counterfactual fidelity, reporting competent cohorts separately from failed diagnostics. H961 context-bridge predictor evaluation remains pending.
