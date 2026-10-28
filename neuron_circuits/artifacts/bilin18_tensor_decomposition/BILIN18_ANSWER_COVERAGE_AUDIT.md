## H760 — Failed-task audit separates conditional discrimination from unconstrained completion

Cached native H732 logits are rescored using symmetric answer-format rules fixed before inspecting predictions in this audit. Three sets are compared: original labels, canonical labels expanded across lower/title/upper case and zero/one leading space, and the existing semantic alternatives with the same case/space expansion. Only exact single-token continuations preserving prompt tokenization are retained; token IDs are deduplicated and classes remain disjoint. No cases are dropped based on predictions. This is posthoc measurement diagnosis, not a new confirmatory task test.

### Sentiment

The original broad answer sets omit capitalized “ Excellent,” although they include lowercase “ excellent.” That token is native top-1 on14/16 format-A endpoints and12/16 format-B endpoints, including negative reviews. Expanding the broad sets therefore raises intended-class top-1 from0% to50%, but both-endpoint choice remains0%: the model still favors the positive semantic bucket at both endpoints. Other top predictions include “ The,” “ Strong” and “ I.”

Canonical positive/negative labels tell a different story. With symmetric case/space variants, both-endpoint binary choice is100% on both formats, while intended full-vocabulary top-1 remains0%. Mean probability mass on the correct canonical class is only0.84%(A)/1.51%(B). Thus the model contains a usable conditional label distinction while its unrestricted next-token distribution does not perform the requested classification reliably. A broad synonym bucket can obscure that distinction because dominant continuation preferences overwhelm the narrow labels.

This does not justify retroactively declaring sentiment competent under the earlier dual criterion. Nor does the earlier failure show that sentiment information is absent. Forced-choice discrimination, answer-format obedience and unconstrained completion are separate measurements. The original causal studies used the original broad margin and remain measurements of that margin; a canonical-label causal test would be a different experiment.

### Capital recall

Format A remains100% correct in paired choice and12.5% intended full-vocabulary top-1 under all scoring rules. Its16 top predictions include eight currency symbols, six country names and two city answers (“ Moscow”). Case/space expansion does not rescue it. Currency predictions suggest that “Capital:” can evoke a financial interpretation, but this audit alone does not prove the cause. Format B, which explicitly asks for the capital city, stays100% on both metrics; its country set also differs, so the comparison does not isolate wording causally.

The appropriate next prompt test should cross explicit city wording with the same countries, and explicit sentiment-label instructions with the same reviews, while keeping scoring rules fixed. The old failures remain visible. A fresh copying/nuisance confirmation should likewise be locked before outputs, with all featurizer seeds frozen and no competence-based individual example filtering.

Audits: original competence metrics replay exactly; class token sets are disjoint; all source/input hashes unchanged. Alternative answer strings failing single-token continuation checks are recorded. Zero model forwards or reserved-test evaluations; natural document pool26,112 unchanged. Multi-token and delayed answers are not assessed here.

Next: design the prospective copying/nuisance panel and a separate prompt/answer-format competence check. Keep canonical forced-choice evidence distinct from full-vocabulary success, and do not silently replace prior causal outcomes with a more favorable score.
