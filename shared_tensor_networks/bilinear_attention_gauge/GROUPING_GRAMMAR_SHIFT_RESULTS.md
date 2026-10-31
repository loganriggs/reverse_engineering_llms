# When unlabeled co-occurrence grouping stops identifying a whole task variable

The [five-axis graph test](GRAPH_GROUPING_RESULTS.md) recovered selectively causal A/B groups from 32 unlabeled toy-language sentences. That grammar always puts exactly one token from each A, B, and filler family in a sentence. Here the trained rank-five sparse dictionaries are **held fixed**, while the unlabeled key bags used for graph grouping change. The trained model is not run on these variable-length bags: its forward method expects the original four-position sentence. This is a stress test of the *group inference statistic*, not a claim about model task accuracy under a new grammar.

Five laws generate key-token **counts** over the same six tokens: exactly one per family; one per family with independent 30% family dropout; zero, one, or two independent draws per family; independent Poisson(.75) counts for each token; and exact one A/B token with Poisson filler counts. For each law, sample 32, 512, or 4,096 unlabeled bags. As before, project counts onto each fixed sparse dictionary axis, connect axes whose absolute activation correlation exceeds .4, .6, or .8, and use connected components as groups. No family labels enter grouping.

The earlier success criterion was **too permissive**: an isolated A0-specific axis can have a very selective A-flip effect, so finding *some* A-dominant and B-dominant groups does not prove that the A0/A1 and B0/B1 axes were reunited. A stronger post-fit criterion requires a group's effect purity to exceed .9 **and** its subspace to cover more than .9 of the complete signed A0↔A1 or B0↔B1 contrast. This full-contrast test is used only for evaluation; it does not influence the graph.

| Unlabeled key-bag law | Mean within-A token-count correlation | Complete A/B groups at threshold .4 | At .6 | At .8 |
|---|---:|---:|---:|---:|
| Exactly one per family | -1.00 | 9/10 | 9/10 | **9/10** |
| Family dropout, presence .7 | -.54 | 9/10 | 0/10 | **0/10** |
| Zero to two draws per family | -.20 | 0/10 | 0/10 | **0/10** |
| Independent Poisson token counts | ~0 | 0/10 | 0/10 | **0/10** |
| Exact A/B, Poisson filler | -1.00 | 9/10 | 8/10 | **6/10** |

These are ten seeds with **4,096** unlabeled bags each. At threshold .8, the complete-group results for 32 bags are 8/10, 0/10, 0/10, 0/10, and 7/10 in the same law order. More samples do not repair the zeroes: the population correlations themselves changed. At threshold .8 and 4,096 bags, dropout, variable-count, and independent-Poisson laws all return five singleton axes in every seed. The old effect-only criterion still reports A/B success in 10/10 seeds for **all** those laws, showing why it was misleading as a measure of grouping.

There is a simple theoretical limit. If a family emits \(N\sim\mathrm{Poisson}(\lambda)\) tokens and each occurrence independently chooses token 0 or 1 with equal probability, Poisson thinning gives independent counts \(C_0,C_1\sim\mathrm{Poisson}(\lambda/2)\). Their covariance is zero. For near-token-specific sparse axes, no co-occurrence correlation identifies that the two axes belong to the same family. Indeed, absent output behavior or intervention evidence, permuting these independently sampled token identities leaves the unlabeled count law unchanged. Family dropout retains negative correlation, but at presence probability .7 its population correlation is \(-p/(2-p)\approx-.538\), below the original .8 graph threshold; lowering the threshold to .4 recovers nine full A/B groupings.

The rank-five sparse basis itself has not lost all A/B task information; the failure is in **grouping from the chosen unlabeled distribution**. This points to a concrete next featurizer requirement: combine co-occurrence with a functional or intervention-derived relation between atoms, and evaluate whether it recovers complete variables under variable-count grammars. The fixed model's key table and known A/B contrasts still make this a toy-level finding; a richer trained language model is needed to test transfer to realistic attention.

Run `dual_grouping_shift.py` after `dual_intervention_sparse_basis.py`. `dual_grouping_shift.json` records all count laws, sample sizes, thresholds, graph groups, post-fit effect purity, full-contrast coverage, and within-family count correlations.
