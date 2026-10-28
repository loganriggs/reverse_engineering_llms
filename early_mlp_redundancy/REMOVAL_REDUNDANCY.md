# What can actually be removed from MLP1?

2026-10-28. This is the user's intended meaning of redundancy: remove parts while preserving
computation. The previous steering experiments answer a different question. This document
corrects the handoff's framing and reports a direct deletion diagnostic, EMLP-D1.

## Historical evidence: narrower than the prose suggested

The historical 46-unit article-cluster experiment used **mean replacement**, not literal zeroing.
It replaced selected products with their empirical averages, preserving a constant contribution.
It scored an average article probability margin at positions whose next token was an article.
Clustering, means and evaluation used the same 64-document sample in that script. Removing the
cluster's variation changed the margin by−.000681, compared with+.119567 for replacing the whole
MLP1 write by its mean. This supports a non-necessary cluster for that statistic; it cannot prove
that all other units duplicate the cluster or that arbitrary large subsets preserve model outputs.
Mean margins can also conceal changes that cancel across examples.

The token-table result answers a replacement question: token-conditioned writes recover96.01%
of MLP1's mean-ablation CE stake on covered positions. It says neither that96% of channels are
removable nor that MLP1 repeats MLP0. The earlier “layer1 redundant with layer0” inference was
explicitly retracted in historical entries848–849 after geometry measurements showed substantial
re-clustering. Those are whole-layer geometry measurements, not an MLP1-only causal attribution.
A token table can implement meaningful new geometry.

The global-gate experiment also does not settle arbitrary deletion: it rejected a particular
selector/certificate at32/128/512 retained products, with context-fitted and shared coefficients.
MLP2's512-native-product retention failures instead show that aggressive pruning can break
coordinated contributions. None of those results substitutes for a literal-removal test of MLP1.

See [historical excerpts](evidence/prior/EXCERPTS_RESULTS.md),
[original mean-replacement source](evidence/prior/article_mlp1_localization_source.py), and
[gate findings](evidence/prior/MLP1_GLOBAL_GATE_FINDINGS_2026-08-28.md).

## D1: fixed random partition, literal deletion

Partition all4,608 native product channels once, using seed2026102901, into four disjoint groups
of1,152. Zero each group separately, the two complementary halves(0+1 and2+3), and all products.
All edits apply at every token position in MLP1; the output bias remains. Include a separate
whole-write-zero control. No ranking, rescaling, optimization or refitting. MLP0 and MLP2 are live.

Evaluate all160 endpoints of the existing80-pair training bank, including incorrect examples and
all ten families. This is an opened synthetic/task panel, not an independent natural-corpus test.
Whole-task evaluation exclusions remain untouched. No arbitrary fidelity threshold is declared
after seeing the results; report continuous metrics and the worst endpoint as well as the mean.

KL is native||edited over the full50,304-output distribution at every prompt position. Top1
agreement compares the edited model with the native model, irrespective of correctness. Next-token
CE uses the actual subsequent prompt tokens, not the task answer. All summaries weight endpoints
equally; the reported range across quarters is not a confidence interval.

| Deleted | Mean KL | Native-top1 agreement | Observed-token ΔCE | Forced-choice task accuracy |
|---|---:|---:|---:|---:|
| None |0|100%|0|98.75%|
| Quarter0 |.00806|94.05%|+.00771|98.75%|
| Quarter1 |.01081|91.55%|−.01869|98.75%|
| Quarter2 |.01100|94.14%|+.01959|98.75%|
| Quarter3 |.01901|94.47%|−.00301|98.75%|
| Half0+1 |.03033|87.58%|−.01556|100%|
| Half2+3 |.04362|89.43%|+.04695|98.125%|
| All products, bias retained |.59793|53.44%|+.45492|85.625%|
| Whole write, bias removed |.60303|55.80%|+.45220|87.5%|

The worst endpoint's mean KL is.035–.048 for quarters and.149–.188 for halves. Average performance
does not guarantee per-input preservation. Task full-vocabulary accuracy and per-family outcomes
are also retained in [the result JSON](validation/mlp1_group_deletion.json).

## What the result supports

MLP1 tolerates substantial literal channel deletion on this panel, especially under the binary
task metric. That is direct evidence of partial robustness to removal. It is not exact functional
equivalence: even quarter deletions change about6–8% of native top predictions across prompt
positions. A lower CE in some arms does not mean the same computation survived.

The whole-product intervention is much more disruptive than either half separately. This is
consistent with overlapping support or downstream compensation, but the nonlinear layer stack
and nonlinear KL metric also produce interactions. It does not establish that the two halves
implement identical subfunctions. Distributed coding, duplicate computation, and tolerance of
small write perturbations remain distinct hypotheses.

The sharper contrast is **binary decisions can survive while the full output distribution changes**.
This is why clustering/patching, token-table replacement, and steering success should not be
summarized as “many units can be removed without changing computation.”

## Next bounded test

Before searching for deletable subsets or resuming the shared-direction project, distinguish
amplitude change from direction damage for these same fixed masks. For each native MLP1 write y
and literally deleted write y_del, construct a direction-preserving control
`y * ||y_del|| / ||y||` per position, with a defined zero-norm guard. Compare its downstream effects
with literal deletion. This control is not itself channel removal and must never be counted as a
pruned model. It tests whether loss of write magnitude explains much of the observed damage.
Retain the same task panel as development data and do not tune masks. A corpus-level removability
claim will require a separate frozen natural-text evaluation and an explicit fidelity tolerance;
D1 does not supply either. The P10 projection/residual follow-up is deferred while answering the
user's removal question, not discarded or silently completed.

## Reproduction and validation

```bash
python probe_mlp1_group_deletion.py --output validation/mlp1_group_deletion.json
python audit_mlp1_group_deletion.py
```

1,441 model forwards. Exact recomputation identity check; all160 native task outputs replay P7
exactly. Partition/arm coverage and aggregate metrics audited. Full-position metrics independently
recomputed in NumPy for eight saved sequence-output pairs, maximum error2.39e-8 (float32 top1
fraction rounding). All-product output equals the native bias exactly; whole-write output equals
zero exactly. The audit expects `/tmp/emlp_d1_deletion_logits.pt` written by the runner, not bundled.
[Audit receipt](validation/mlp1_group_deletion_audit.json).
