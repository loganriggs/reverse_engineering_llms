## H882 — Three additional competent domains support a broader mechanistic transfer test

The previous Jacobian experiments focused on matched copying and capital recall. H882 predeclared eight task domains, two templates each, and a matched repeated-token copying control for every source contrast. Subject–verb agreement, country-to-language recall, and human-name binding pass the fixed competence gate together with their copying controls in both templates. These supply 192 contexts for a prospective partner-transfer test. The screen establishes behavioral eligibility, not transfer of the earlier mechanism.

### Design and gate

Eight domains: weekday succession, month succession, subject–verb agreement, antonyms, past tense, plural forms, country-to-language recall, and human-name binding. Total 63 source pairs and 504 contexts (two templates × task/copy × two endpoints per pair). Each endpoint differs only at aligned source-token positions, and those leading-space source tokens are identical between the task and copying contexts. Sources and candidate answers must be single tokens, candidate answers distinct, and both endpoints equally long within each task/template. All pairs passed lexical eligibility; none were removed based on correctness.

The gate, frozen before execution, requires at least four eligible pairs in a task/template cell, both endpoints binary-correct on at least 75% of pairs, and pooled full-vocabulary top-1 accuracy at least 50%. Both task and its same-template copying control must pass. Every failed cell and native prediction is retained. These are fresh combinations and task contrasts relative to H873, not necessarily globally unseen words/templates or unseen pretraining examples. Prior broad task screens already included related task families.

### Outcomes

Entries below give task full-vocabulary accuracy, with paired eligibility requiring the copying control as well. Binary correctness is judged against the declared two candidate answers; full-vocabulary correctness requires the intended token to outrank all other tokens.

|Domain|Pairs|Task top-1 A|Task top-1 B|Both-endpoint binary A|Both-endpoint binary B|Eligible matched formats|
|---|---:|---:|---:|---:|---:|---|
|weekday_successor|7|0.000|0.000|1.000|1.000|None|
|month_successor|12|0.000|0.000|1.000|1.000|None|
|agreement|8|0.750|0.938|1.000|1.000|A,B|
|antonym|8|0.688|0.000|0.875|0.000|None|
|past_tense|6|0.000|0.000|0.667|0.667|None|
|plural|6|0.000|0.000|1.000|1.000|None|
|language|8|0.938|1.000|1.000|1.000|A,B|
|binding|8|0.875|1.000|1.000|1.000|A,B|

Agreement achieves 75–94% full-vocabulary task accuracy, language recall 94–100%, and binding 88–100%, each with 100% both-endpoint binary accuracy. Their copying controls also pass. These cover grammatical number, a different factual association, and person-role binding, providing a more diverse next test than additional capital-recall variants.

Antonym template A passes as a task, but its matched copying control fails the paired binary gate (62.5%). Template B has the opposite problem: copying passes but the antonym task fails. Neither task/copy pairing is eligible; mixing templates after seeing results would change the experiment and is not done here.

Weekday and month task prompts always predict “the” as the immediate next token, despite 100% binary discrimination between the declared answer candidates. This is a continuation-format failure under this exact next-token gate, not proof that the model lacks calendar knowledge. Plural and tense prompts also fail the declared gate. No templates are repaired after observing results in this experiment.

### Frozen next intervention

Before measuring mechanistic effects on these eligible cells, saved the H883 design: retain all 192 agreement/language/binding contexts, both templates, and the previously frozen L8H3 reader. Compare live and recipient-native linearized MLPs with normalization-only, partner-only, and full-factor sharing, in both copying-donor and task-donor directions. Keep attention live, recipient offsets native, and use the same source-bus difference with matched common amplitude. Do not choose a stronger reader after seeing these effects.

Measure source-token and task-answer output contrasts under native and shared-scale uncapped readouts. Include individual signed effects, absolute response strength and vocabulary-wide native KL. Keep tiny/zero routing amplitudes explicit rather than filtering them based on effect quality. Weak effects would limit what this reader can establish on a task; competence alone does not guarantee that L8H3 carries its computation. The prospective comparison distinguishes transfer of the partner mechanism from a result specific to country/capital recall.

### Verification and limits

516 full-model forwards, including the replay batch, over 504 unique panel contexts. Native forward replay is exact. Signed attention-times-values reconstructs captured head outputs within relative 1.9e-06; same-token persistent-bus vectors agree within max absolute 1.53e-05. Independently rechecked endpoint alignment and task/copy source-token equality for all 504 contexts. All hashes unchanged and outputs finite. No reserved test evaluated; the natural-document pool stays at 26,112. Generated synthetic prompts do not increase that document count. The broad research goal remains active.

No new featurizer or manifold has been validated by this screen. The positive result is a frozen, diverse behavioral panel on which the next causal comparison can be interpreted without relying on binary answer preferences alone.
