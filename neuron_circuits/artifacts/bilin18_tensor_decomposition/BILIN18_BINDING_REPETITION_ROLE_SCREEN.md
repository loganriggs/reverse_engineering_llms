## H922 — Explicit-fact queries reveal an event-order shortcut, not reliable semantic role switching

H921 showed useful transfer of signed occurrence coefficients in the original incomplete-sentence panel. H922 tests a limitation of that account: whether routing adapts when the more-repeated name is the requested answer. The fixed screen independently crosses name-list order, sender identity, extra-mentioned identity, sent-to versus received-from event wording, and sender versus recipient question.

Example: “Yesterday, John and Mary met. John sent a letter to Mary. John was smiling. Question: Who sent the letter? Answer:” and the same context asking who received it. Received-from reverses event argument order while preserving who sent to whom. This is a lexical argument-order reversal, not grammatical passive voice. Each name occurs once in the list and once in the event; the extra sentence gives one name a third occurrence. All256 rows are retained, with no prompt changes after results.

### Competence

Each cell has16 endpoints from four pairs, two list orders and two sender identities. Binary competence requires both sender-identity endpoints correct within each pair/order. The prior .75 paired-binary and .50 exact-top1 thresholds apply.

|Split|Event wording|Question|Extra-mentioned name is answer|Both-binary|Top1|Pass|
|---|---|---|---|---:|---:|---|
|discovery|sent_to|sender|True|0.000|0.250|False|
|discovery|sent_to|recipient|False|0.625|0.812|False|
|discovery|received_from|sender|True|0.250|0.625|False|
|discovery|received_from|recipient|False|0.000|0.438|False|
|discovery|sent_to|sender|False|0.000|0.375|False|
|discovery|sent_to|recipient|True|0.125|0.562|False|
|discovery|received_from|sender|False|0.250|0.625|False|
|discovery|received_from|recipient|True|0.000|0.125|False|
|transfer|sent_to|sender|True|0.000|0.000|False|
|transfer|sent_to|recipient|False|1.000|1.000|True|
|transfer|received_from|sender|True|0.875|0.938|True|
|transfer|received_from|recipient|False|0.000|0.125|False|
|transfer|sent_to|sender|False|0.000|0.125|False|
|transfer|sent_to|recipient|True|0.875|0.938|True|
|transfer|received_from|sender|False|0.875|0.938|True|
|transfer|received_from|recipient|True|0.000|0.000|False|

No discovery cell passes. Four transfer cells pass, but they align with the second event argument: recipient in sent-to wording, sender in received-from wording. Both questions are not reliably answered for the same facts. These selected successes therefore cannot justify a claim of question-sensitive role binding.

### Shortcut diagnostic

Binary preference below compares the two names, independently of the requested answer. “Both queries correct” requires the preference to change appropriately when the question changes, holding all facts fixed.

|Split|Wording|Prefer second event name|Prefer extra-mentioned name|Both queries correct|Same preference across queries|
|---|---|---:|---:|---:|---:|
|discovery|sent_to|0.688|0.406|0.062|0.875|
|discovery|received_from|0.672|0.422|0.031|0.844|
|transfer|sent_to|0.953|0.453|0.031|0.969|
|transfer|received_from|0.938|0.469|0.031|0.938|

On transfer names the model prefers the second event name in95.3% of sent-to and93.75% of received-from rows. Both questions are correct for only3.125% of context pairs. The same name remains preferred across question changes in96.875% and93.75%, respectively. Extra-name preference stays near chance. This rules out interpreting high accuracy in isolated cells as general role-sensitive behavior; event-order preference is a plausible explanation requiring causal testing.

Question changes do affect the output distribution (mean query-pair KL about.014–.017), so the model is not literally invariant to the query. Nor does event-order correlation establish a causal mechanism or prove the model lacks semantic information. It describes the behavior of these prompts.

### Implications and next test

Do not extend a prompt-rescue search or fit a semantic routing model on only the passing cells. H921's shared occurrence computation remains supported within its competent original templates, but extension to explicit question answering is unverified. The three-name-occurrence rule cannot be directly transplanted into this five-occurrence format without specifying a new mapping.

Use this panel as a negative control with an observable alternative behavior. Test local persistent-value contributions from each of the five name occurrences at the fixed readers, using the same causal method as H920. Evaluate effects on event-second versus event-first name preference regardless of requested answer, alongside task correctness, name log probabilities and full-distribution KL. Compare event-name ablations with the final extra mention and initial list. This can determine whether the model reuses lexical promotion machinery to implement an event-order shortcut. Report all query conditions; successful causal explanation of a shortcut is not successful semantic binding.

### Verification

256 native forwards, all source hashes unchanged and measurements finite. Token IDs identify exactly three occurrences of the extra-mentioned name and two of the other name; continuation tokens are single-token. All competence outcomes, native logits, persistent values and routing vectors are saved. No interventions or operator fitting were performed. No reserved test or added natural documents; pool remains26,112. Broad goal active.
