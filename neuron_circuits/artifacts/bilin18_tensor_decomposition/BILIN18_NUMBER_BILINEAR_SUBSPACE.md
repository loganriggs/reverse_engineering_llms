## H593 — A simple context-by-number interaction does not preserve the tested causal context switch

Previous turn made progress by identifying the local validity range of MLP11's linear-in-response action. For a context change dc and fixed normalized-input perturbation e, the change in MLP response is exactly T(dc,e)=Down[Left(dc)Right(e)+Left(e)Right(dc)]. The quadratic-in-e term cancels. This permits an explicit test of a compact context-by-signal computation.

Fit separate uncentered SVD bases to 32 regular/fronted context differences (unfronted minus fronted) and their positive small-dose input responses. These are eight lexical pairs crossed with subject/attractor number, not 32 independent nouns. Freeze before interventions. Test projecting context only, incoming response only, or both at ranks1/4/8. A rank-one-both model has the form (a dot dc)(b dot e)T(a,b): one scalar-times-scalar output vector. Actual donor/context coefficients are still needed.

Replace only the context-switch contribution around each recipient's native MLP response. Full-context-switch and unchanged-context controls define fidelity. Five random rank-one context/input pairs are separately norm-matched to learned projected inputs. Evaluate irregular nouns and relative-clause contexts held out from basis fitting, plus both dose signs and a larger dose. All prompts were explored previously; this is not an untouched confirmatory corpus. Total 8,192 interventions.

| Small-dose cell | Rank1 both: relative margin error | Rank8 both: relative margin error | Rank8 context only: relative margin error | Rank8 incoming only: relative margin error |
|---|---:|---:|---:|---:|
|regular/fronted_preposition|0.7839|0.5525|0.5393|0.0521|
|regular/relative_clause|1.0465|1.0134|1.0099|0.2232|
|irregular/fronted_preposition|0.8120|0.6805|0.6885|0.1610|
|irregular/relative_clause|0.9728|0.9514|0.9646|0.3714|

The rank-one interaction is not a faithful account of this context-switch effect. On irregular fronted prompts its relative margin error is 0.812 and relative reference KL 0.781; rank eight still has 0.681/0.599. Relative-clause transfer is poorer. These errors are relative to the full context-switch versus unchanged-context discrepancy, not to the whole number perturbation or task accuracy.

The larger bottleneck is contextual representation: keeping full context and compressing only incoming response works better than the reverse. On regular fronted discovery rows, rank-eight incoming-only margin error is 0.052, versus context-only 0.539. Context rank eight retains 91.6% of discovery context energy, yet does not retain comparable causal fidelity. Raw variance preservation is insufficient. Random pair controls generally remain near unchanged-context error, and all seeds are retained.

This does not prove high intrinsic dimension or rule out a useful bilinear unit. It may reflect a poor projection objective, limited discovery span, or variation in which context directions matter. Next H594 uses all 32 stored discovery directions without refitting. This tests whether the rank cutoff alone explains failure; it is still an orthogonal projection, not an optimal response-space oracle.

Audits: exact context-switch identity <=1.87e-6 relative error; matched input norms <=2.37e-7 relative error; full/unchanged references replay H591 within 1.44e-5 margin; finite metrics, unchanged source and both frozen bases. KL/group NLL use float64 from float32 model logits. No new prompts or natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
