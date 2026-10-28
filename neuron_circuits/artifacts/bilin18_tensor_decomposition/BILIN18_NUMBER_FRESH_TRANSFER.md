## H597 — Frozen-basis advantages persist on new lexical/wording combinations, but ordinary projection remains weak

Freeze H596's old32/new32/new64 context spaces and H584's number direction before inference. Construct 384 prompts crossing eight new regular subject pairs and eight new irregular pairs, new attractors, both number factors, three sentence frames, and familiar versus new wording. Familiar wording uses beside/can see; new wording uses across from/might notice. Each context-switch recipient uses its unfronted anchor within the same wording condition. The new irregular set includes ox/oxen, louse/lice, cactus/cacti, fungus/fungi, nucleus/nuclei, analysis/analyses, criterion/criteria and phenomenon/phenomena.

No prompt string overlaps the enumerated H586/H587/H582 manifests. This means new combinations within that checked sequence, not training-data novelty or exhaustive global-history novelty. Baseline captures at native and signed small dose use 1,152 model forwards. Test full/unchanged references plus projection/oracle in all three spaces: 4,096 intervention forwards. No refitting or performance filtering.

All 24 type/frame/attractor-number competence cells pass the predeclared gate. Candidate both-endpoint accuracy is 1.000 except one cell at 0.875; full-vocabulary top1 ranges 0.500–1.000. Some irregular fronted cells are close to the gate, so passing does not mean every prompt is correct. All rows remain.

| Fresh cell | New64 local oracle error | Projection relative margin error | Oracle relative margin error | Oracle relative reference KL |
|---|---:|---:|---:|---:|
|regular/fronted_old|0.6483|0.8291|0.1861|0.2609|
|regular/relative_old|0.4713|1.0156|0.2346|0.1250|
|regular/fronted_new|0.7313|0.8062|0.3086|0.3871|
|regular/relative_new|0.6362|1.0095|0.0961|0.1829|
|irregular/fronted_old|0.6102|0.4546|0.0840|0.0436|
|irregular/relative_old|0.4776|0.6859|0.1621|0.0858|
|irregular/fronted_new|0.6734|0.6853|0.1225|0.1161|
|irregular/relative_new|0.6514|1.3273|0.1002|0.0768|

The broader context span retains an advantage. For familiar-wording relative clauses, old32-to-new32 oracle local error drops 0.877 to 0.534 on regular nouns and 0.842 to 0.532 on irregular nouns. With new wording the improvement is smaller. New64 improves further, but substantial internal error remains, and output fidelity is not uniform. Ordinary projection remains especially poor for some relative-clause margins despite better KL.

This is a fresh frozen-basis test, but oracle coefficients still use each case's full local target response. It does not demonstrate a target-free representation. The next follow-up tests a fixed projection metric trained solely on the old regular discovery input responses. That follow-up is motivated by these outcomes and must be labeled adaptive, even though fitting will exclude all fresh cases.

Audits: no basis refit, zero checked-manifest overlap, locked manifest/source/bases unchanged; native-response replay <=1.24e-5; exact context-switch identity <=2.04e-6 relative error; oracle normal equations <=8.73e-15; finite float64 metrics. No new natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
