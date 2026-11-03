# V11 section topic enters the joint narrative distribution

V10 already samples narrative verb/object mixtures, source claims and questions, section facts and questions, and bracket spans in one stream. But the section heading's **topic** was inert for ordinary narrative predictions. V11 adds one explicitly shared computation: the latest completed visible `Section N: TOPIC.` heading changes the next place distribution in later ordinary prose. For example, `Section 3: school.` makes `school` more likely in a subsequent `In the ...` or `near the ...` place slot, when that word is grammatical there. A place generated this way also enters the ordinary narrative history and can be asked about by the existing `Where` question. The other V10 rules are unchanged. Source: [`surface_mixture_v11_section_topic.py`](surface_mixture_v11_section_topic.py).

At an eligible place slot, with V10's exact baseline distribution \(p_0\), current heading topic \(h\), and configured mass \(m=.35\), the new oracle is

\[
p_{11}(w\mid x_{<t})=(1-m)p_0(w\mid x_{<t})+m\mathbf 1[w=h].
\]

If no completed heading exists, or its topic is unavailable at this grammatical slot, V11 exactly retains V10. The heading and any intervening tokens are visible; there is no hidden scheduled task. The [`audit`](audit_v11_section_topic.py) reconstructs the full place distribution from named baseline and topic components, checks every sampled token's support, and changes only a heading topic while preserving the complete intervening prefix in support. Setting \(m=0\) replays V10 sequences and next-token distributions exactly on 32 additional streams per audit.

| 400×256-token panel | Reference seed 98401 | Reference seed 98411 | Enriched-training seed 98421 |
|---|---:|---:|---:|
| Streams with heading-influenced place site | 47 | 62 | 141 |
| Heading-influenced place sites | 244 | 302 | 690 |
| Streams with verb, object, topic-place, and bracket | 22 | 30 | 59 |
| Supported heading-topic edits checked | 244 | 256 | 256 |

The first two panels use the same *provisional reference candidate* rates as V10, including `section_rate=.06`. The third uses `section_rate=.18` only for training enrichment. This is not a natural-frequency estimate. The selected heading→place distances range from 4 to over 220 tokens. Empirical topic matches varied across small panels (.348/.295/.326 versus oracle .364), so a further 2,000-stream reference calibration checked 1,524 eligible sites: empirical .3655 versus oracle .3638; the V10 baseline at those sites averages .0212. The larger panel supports sampling variation as the explanation for the small-panel shortfall. Both the component arithmetic and the zero-mass V10 replay have maximum recorded error zero in the checked float representation. Raw panels: [reference A](results/v11_section_topic_audit_seed98401_n400.json), [reference B](results/v11_section_topic_audit_seed98411_n400.json), [third reference](results/v11_section_topic_audit_seed98431_n400.json), [enriched training](results/v11_section_topic_conditional_audit_seed98421_n400.json), and [large calibration](results/v11_section_topic_calibration_seed98701_n2000.json).

The FineWeb check is **negative**. On an already-used 192-window panel, V11 and V10 have the same measured first-40-word form statistics: question-mark windows .479 versus FineWeb .234, colon windows .833 versus .432, parentheses .302 versus .490, and repeated-word fraction .272 versus .164. Enriching section starts for training raises colon windows further to .896. These are diagnostic form comparisons with different tokenizers, not calibrated dependency frequencies. In the separate manually reviewed 32-window FineWeb development panel, 10 windows had some scope relation, but the visible examples include policy and procedure headings, testimonial groups, numbered records, and FAQ sections. None of the inspected short excerpts is a close instance of this exact place-heading rule. V11 is therefore a **mechanistic proxy for topic scope**, not a verified natural-language DGP. See [reference](results/v11_section_topic_fineweb_form_seed98601.json) and [conditional-training](results/v11_section_topic_fineweb_form_conditional_seed98601.json) form results and the earlier [FineWeb dependency review](fineweb_dependency_second_panel.json).

The model test uses the already-trained V10 two-layer, four-head, width-64 bilinear transformer as a baseline, then continues training on the V11 joint stream with the enriched section rate. The reference-rate evaluator compares exact-oracle KL at heading-influenced place sites and a supported heading-only edit that should move probability toward the new place by exactly .35. It also reports the previously learned verb, object, bracket, source, and section-answer rules separately. The prospective competence gate is reference-rate topic-site KL below .1 nat, mean edit response within .05 of .35, and no >.1-nat KL regression on any previously competent verb/object/bracket family. Failure on any condition calls for a training or DGP adjustment before interpreting a section-topic circuit. Training and [evaluation](eval_v11_section_topic_teacher.py) are deliberately task-wise: competence on the new place rule cannot be inferred from overall language loss, and no source/section-answer interpretation is licensed if those rules remain weak.

The ordinary-CE continuation ran 800 updates with batch 32×256, RMSNorm, learning rate .001, and initial V10 weights. It was trained at `section_rate=.18`; all results below use fresh *reference-rate* V11 streams (`section_rate=.06`) and 256 randomly selected supported heading edits per panel.

| Metric | V10 pretrained, panel 98501/98511 | V11 continuation, panel 98501/98511 |
|---|---:|---:|
| Topic-place exact-oracle KL | .818/.824 | .585/.574 |
| Heading-edit new-topic response | .00001/~0 | .00002/.00002 |
| Exact heading-edit oracle response | .35/.35 | .35/.35 |
| Full edit-effect L1 error | .705/.705 | .706/.705 |
| Verb KL | .524/.521 | .462/.457 |
| Object KL | 1.276/1.271 | 1.184/1.175 |
| Bracket KL | .110/.119 | .088/.091 |

The topic-site KL improved, but the causal response stayed at zero, including near, middle, and far heading distances. The model learned easier statistics of place sites rather than the heading dependency. Other measured rules did not regress, but the prospective topic competence gate **failed**. Source answers remain around 2.1 nats KL; the very rare section-answer sites are too few to infer reliable learning. Do not decompose or sparsify a purported section-topic circuit in this checkpoint. [Training record](results/v11_section_topic_depth2_from_v10_seed1_800/results.json), [reference panel A](results/v11_section_topic_fresh_seed98501.json), and [reference panel B](results/v11_section_topic_fresh_seed98511.json) preserve the numerical evidence.

A matched 300-step continuation from that *same* checkpoint tested whether explicit exact-oracle supervision fixes the missing dependency. Both arms used batch 32×256, seed 2, learning rate .0005, enriched training section rate .18, and a weight-1 soft target on ordinary topic-place sites. The second arm additionally used a weight-1 full-oracle target for supported heading-topic edits generated from the training batch. Evaluation used two **new** reference-rate panels, each with 512 fresh streams and 256 supported heading edits; the replacement topic was randomly chosen among eligible alternatives rather than fixed to the first alternative.

| Metric | 800-step start, panels 98521/98531 | Soft target only | Soft + heading edits |
|---|---:|---:|---:|
| Topic-place oracle KL | .583/.587 | .435/.448 | .503/.506 |
| Heading-edit new-topic response | ~0/~0 | .011/.015 | .0015/.0017 |
| Exact-oracle response | .35/.35 | .35/.35 | .35/.35 |
| Full edit-effect L1 error | .704/.705 | .706/.705 | .713/.716 |
| Verb KL | — | .455/.457 | .451/.454 |
| Object KL | — | 1.171/1.163 | 1.164/1.159 |
| Bracket KL | — | .106/.100 | .071/.067 |

The extra heading-edit loss **worsened** the measured heading response and topic-place KL relative to the soft-only control. Its near, middle, and far edit responses remain at or below .004. This is a one-seed, one-coefficient result, not a conclusion that causal supervision cannot work. It is enough to stop treating either checkpoint as a learned heading-topic circuit. The model may need a shorter or more frequent dependency, different architecture/curriculum, or a better DGP with broader naturalistic uses of headings. A prospective next screen should require edit fidelity before any sparse-weight graph or canonicalization claim. [Soft-only training](results/v11_section_topic_softonly_seed2_300/results.json), [causal training](results/v11_section_topic_softcausal_seed2_300/results.json), [soft-only panel A](results/v11_section_topic_softonly_fresh_seed98521.json), [soft-only panel B](results/v11_section_topic_softonly_fresh_seed98531.json), [causal panel A](results/v11_section_topic_softcausal_fresh_seed98521.json), and [causal panel B](results/v11_section_topic_softcausal_fresh_seed98531.json) preserve the comparison.

The follow-up should replace the narrow place-heading proxy with a richer heading-to-content relation informed by actual web-text forms, while retaining this version as a causally auditable control. For interpretability, compare whether a weight-only or data-weighted decomposition identifies the heading-topic path and preserves the other competent rules on untouched streams; evaluate reconstruction, task oracle KL, heading-edit fidelity, and unlabeled new-task discovery separately. A rule label used to define the edit is not evidence of unsupervised discovery.
