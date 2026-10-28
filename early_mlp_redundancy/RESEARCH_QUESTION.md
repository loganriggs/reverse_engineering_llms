# Active question: removability of MLP1

The user's definition is: can many parts be removed while the model performs the same computation?
MLP1 is zero-based block1. Separate four claims: a clustered signal can be patched; a module can
be approximated by a token table; a direction can steer behavior; a subset can be deleted without
materially changing outputs. Only the last directly addresses removal redundancy.

Read [REMOVAL_REDUNDANCY.md](REMOVAL_REDUNDANCY.md) for the source-level history correction,
fixed literal-deletion experiment, limitations and next comparison. Read the README for loading.

The primary outcome is native-versus-edited output distribution, supplemented by task accuracy,
observed-token CE, worst-case examples and local write changes. A preserved binary decision alone
is insufficient. A distributional tolerance must be explicit before making an approximate-equivalence
claim; this diagnostic reports effects without inventing a passing tolerance after the fact.

Next: compare the fixed literal deletions with per-position controls that preserve native MLP1
write direction while matching the deleted write's norm. Do not count that control as pruning.
This separates amplitude effects from direction loss. Use the same opened task bank as development,
no mask search or refitting. Generalization to natural text remains a separate required test.

The previous [context-response plan](ARCHIVED_CONTEXT_RESPONSE_PLAN.md) is retained as history.
Its proposals and P10's projection/residual test are deferred, not additional active next steps.
The broader goal remains to identify faithful computational units and eventually use diverse
competent-task causal targets to improve featurizers, preserving whole-task evaluation exclusions.
