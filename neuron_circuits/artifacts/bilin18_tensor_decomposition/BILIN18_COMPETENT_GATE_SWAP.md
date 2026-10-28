## H678 — One coefficient switches month copying to succession, asymmetrically

H677 implicated query-head4 writes in competent plural/month prompts. H678 swaps only the signed query-head4 attention coefficient between paired task/copy prompts, keeping the receiver effective value and every other attention contribution intact. Full downstream computation follows. Three prequalified single-token cells are used: plural arrow, plural labeled, successor arrow; eight items and both instruction directions,48 prompts. All items are retained, including native errors.

A fixed cyclic next-item donor under the same instruction is a separate control. It is not matched in coefficient change magnitude and is not a null distribution. Opposite-instruction prompts differ in demonstrations as well as instruction text. No isolated instruction-token claim is made.

| Family | Format | Receiver | Donor | Correct-target ΔCE | Task−copy margin change | Native KL | Receiver correct/8 |
|---|---|---|---|---:|---:|---:|---:|
|plural|arrow|copy|baseline|+0.0000|+0.0000|+0.0000|7|
|plural|arrow|copy|opposite|+0.0179|-0.0084|+0.0005|7|
|plural|arrow|copy|same_instruction_other_item|-0.0016|+0.0068|+0.0010|7|
|plural|arrow|task|baseline|+0.0000|+0.0000|+0.0000|7|
|plural|arrow|task|opposite|-0.0257|-0.0102|+0.0008|7|
|plural|arrow|task|same_instruction_other_item|+0.0197|-0.0031|+0.0027|7|
|plural|labeled|copy|baseline|+0.0000|+0.0000|+0.0000|8|
|plural|labeled|copy|opposite|+0.0091|-0.0114|+0.0005|8|
|plural|labeled|copy|same_instruction_other_item|+0.0010|-0.0011|+0.0001|8|
|plural|labeled|task|baseline|+0.0000|+0.0000|+0.0000|6|
|plural|labeled|task|opposite|-0.0185|-0.0145|+0.0011|6|
|plural|labeled|task|same_instruction_other_item|-0.0000|+0.0003|+0.0001|6|
|successor|arrow|copy|baseline|+0.0000|+0.0000|+0.0000|8|
|successor|arrow|copy|opposite|+1.4287|+2.4465|+0.6574|0|
|successor|arrow|copy|same_instruction_other_item|+0.0015|-0.0033|+0.0014|8|
|successor|arrow|task|baseline|+0.0000|+0.0000|+0.0000|7|
|successor|arrow|task|opposite|+0.2237|-2.2184|+0.1210|7|
|successor|arrow|task|same_instruction_other_item|+0.0024|-0.0030|+0.0004|7|

On month copying prompts, replacing the query coefficient with its paired successor-task value changes all8/8 full-vocabulary top predictions from the copied month to the correct next month. This includes December→January. Mean task-minus-copy logit margin rises2.4465. This is a behavioral switch, not merely damage to copying or a restricted-candidate effect. Same-instruction cyclic donor swaps preserve8/8 copying and have nearzero mean margin change (−0.0033).

In the reverse direction, replacing the successor-task coefficient with the copy coefficient lowers the task-minus-copy margin in all8/8 examples (mean−2.2184), but task top1 remains7/8. Thus this coefficient is sufficient to induce succession in these copying contexts, but its task value is not necessary for top1 succession in the native task contexts. Other context-dependent computation already supports the task. Copy→task swaps do not establish that the coefficient alone implements or uniquely represents the full successor relation.

Plural swaps are qualitatively different: no accuracy switches, small margin changes, and giving copying prompts task coefficients does not consistently favor plurals. Task prompts receiving copying coefficients slightly improve mean correct-target CE even while their plural-minus-singular margin falls. This again separates task loss from the contrast between two selected candidates, and disfavors treating this scalar as a universal transformation gate.

Leading interpretation: a scalar controls access to an input-associated direction which, in these month contexts, can shift behavior from copying to the next-month continuation. A competing explanation remains that suppressing copying unmasks a preexisting sequential prior, rather than the scalar representing the requested operation. Earlier broad-vocabulary probes showed that this value direction was not uniquely next-month selective, so the current result does not erase that alternative.

Next distinguish operation-specific routing from a released prior with a predeclared next/previous/copy demonstration panel using matched month inputs and formats. Freeze the same gate intervention procedure, inspect native competence first, retain all outcomes, and test whether previous-task coefficients can induce previous-month outputs or merely alter copying/next-month bias. Include another ordered domain before claiming generality. No new arbitrary prompt tuning after intervention outcomes.

Audits:144 whole-model forwards (48×3 conditions). Native replay maximum absolute logit difference2.456e-5. Coefficient×receiver-value write reconstructs H677 saved query-head4 write with maximum relative error1.917e-7. All six tracked hashes unchanged, finite logits. Captures retain receiver values, donor indices and output logits. FineWeb remains26,112 unique documents. Previous turn experimental progress. Broad goal active and incomplete.
