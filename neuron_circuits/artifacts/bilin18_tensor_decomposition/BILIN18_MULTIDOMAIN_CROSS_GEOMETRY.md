## H904 — Stable, compressible directions need not track the task-distinguishing changes

The previous turn made progress: H903 found that retaining group means did not give a reliable advantage over nearby-capacity flat approximations for the fixed number component. H904 broadens the diagnostic to existing language and binding tasks, retaining agreement as a reference.

Language examples ask for the main language spoken in a country (for example Italy/Japan). Binding examples ask for the recipient in a two-name sentence (John and Mary, with the sender changed). These differ from a universal binary number label: endpoint0/1 is arbitrary across country/name pairs and must not be treated as a shared semantic class.

### Fixed discovery/transfer design

Use cached normalized L12 pre-MLP activations from the earlier competent-domain panel. For each domain and template, first four pairs (eight task endpoints) fit the leading centered input PC. Origin is the discovery mean projected on that axis. Fit output PCs of its exact contextual cross-write C(w) using the same discovery endpoints. Last four pairs are disjoint transfer cases, evaluated in both templates and matched copying. No new layer or PC selection, natural evaluation, live intervention, or reserved-test evaluation occurs.

Two distinct metrics: input-axis capture is the fraction of each paired endpoint-difference vector's squared norm projected onto the axis, averaged over four transfer pairs. Output error is relative RMS reconstruction error for the native cross term tC(w), normalized by its removal error. Small output error does not show that the axis represents the task or that the component has meaningful downstream influence.

### Same-template task transfer

|Domain/fit|Mean paired-change energy captured by input axis|Mean-only output error|Rank1 output error|Rank7 output error|
|---|---:|---:|---:|---:|
|agreement_A|0.753|0.254|0.200|0.173|
|agreement_B|0.418|0.497|0.405|0.361|
|language_A|0.044|0.143|0.132|0.121|
|language_B|0.038|0.181|0.170|0.159|
|binding_A|0.042|0.176|0.158|0.143|
|binding_B|0.003|0.164|0.160|0.146|

Agreement's leading axes capture75.3%/41.8% of the corresponding same-template number contrast. Language captures4.4%/3.8%; binding4.2%/0.3%. Yet language/binding mean-only contextual-write errors are already0.143–0.181, with modest improvements from rank1/7. Thus easy approximation of a chosen component is not sufficient evidence that it captures task-distinguishing variation. This does not establish that the remaining small projection is causally irrelevant; output sensitivity has not been measured here.

### Stability and transfer boundaries

Across templates, language input axes have absolute cosine0.955 and output PCs0.896, compared with agreement0.729/0.748 and binding0.714/0.448. Language's relatively stable directions still capture little paired source-change energy. Stability is therefore a separate criterion from relevance to the controlled contrast.

Cross-template rank7 contextual-write error rises to0.277–0.308 in language and0.661–0.771 in binding. On matched copying, language rank7 errors are1.433–1.502 and binding1.124–1.328: worse locally than removing the component. These are local vector errors, not downstream KL or performance effects, and should not be conflated with the earlier H885 Jacobian-response metrics.

The chosen L12 layer came from the number investigation, not a claim that it is optimal for language or binding. All discovery panels are tiny, reused and deliberately structured. The earlier task competence screen supports examining these domains, but this cached geometry experiment does not remeasure live task competence or prove causal relevance of the new axes.

### Next experiment and scope

Compare pooled activation PCs against subspaces fitted to paired source-change vectors, using discovery pairs only. Retain ranks1/2/4 and evaluate on disjoint pairs without treating endpoint order as a common semantic label. Then use each recipient's actual matched-source change for finite interventions, comparing its subspace projection with the full paired change and matched copying controls. This addresses whether extraction is missing the relevant variation rather than merely finding compressible background structure.

Pair construction is experimental supervision, not automatically discovered task labels. A successful contrast-conditioned subspace would still require causal selectivity, stability, novel combinations, and an unlabeled recovery method before being called a fundamental unit. The broader BSF/shared-dictionary/manifold objective remains open; number is no longer the sole test case.

### Verification

Source/model/capture hashes unchanged. Bases orthonormal to5.55e−15; compiled contextual encoders agree with explicit projection to relative9.48e−16. All metrics finite. Six frozen fits saved, each with eight discovery endpoints; transfer pairs excluded from fitting. Zero new full-model forwards, zero reserved-test evaluations, and no natural-document pool increase beyond26,112. Goal active.
