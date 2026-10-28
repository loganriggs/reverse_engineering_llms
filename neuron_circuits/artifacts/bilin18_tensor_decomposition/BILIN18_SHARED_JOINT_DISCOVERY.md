## H700 — Joint label-free learning collapses to constant groups

Fit shared dictionary, linear coordinate encoder,Top3membership logits and quadratic-energy gate detector together from fresh random weights. Use the H692/H693 data split: singleton/pair groups for training/validation, higher combinations held out. For unknown gates, train exact Bernoulli expected reconstruction-plus-group-count cost over16patterns; for unknown memberships, use hard-forward straight-throughTop3. Three seeds,3000steps, same minibatches. Select hard validation reconstruction-plus-count objective every100steps. The joint arm uses no teacher labels in training or selection; teacher dictionary/membership/gates enter only posthoc evaluation. Known latent/group counts and membership size remain supplied hyperparameters.

Run a factorial information control: both true membership/gates supplied, membership only, gates only, neither. Effective trained parameters differ, so this isolates information/optimization bottlenecks rather than resource-matched superiority.

| Seed | Supplied information | Held relative MSE | Mean active groups | Mean active atoms |
|---|---|---:|---:|---:|
|51|oracle_both|1.48735e-11|3.191|7.191|
|51|oracle_membership|5.95051e-05|3.996|7.996|
|51|oracle_gates|0.11753|3.191|6.610|
|51|joint|0.11926|3.000|7.000|
|52|oracle_both|2.68158e-11|3.191|7.191|
|52|oracle_membership|2.10886e-08|4.000|8.000|
|52|oracle_gates|2.1857e-05|3.191|7.191|
|52|joint|3.26602e-09|3.000|8.000|
|53|oracle_both|2.69908e-11|3.191|7.191|
|53|oracle_membership|1.62801e-08|4.000|8.000|
|53|oracle_gates|0.11013|3.191|6.378|
|53|joint|0.110834|3.000|7.000|

The joint arm fails structural discovery in all three seeds. An explicit occupancy audit finds the identical gate pattern[1,1,0,1]on every one of4096training,2048validation and2048heldout inputs in each seed. Thus approximately19%held label agreement reflects the fixed pattern's prevalence, not useful input-dependent inference. After posthoc atom and group alignment, membership-entry agreement is0.75and atom cosine0.609–0.732. These alignment scores do not rescue the constant-gate behavior.

Seed52 nevertheless achieves held relative MSE3.27e-9: its three constant groups expose all eight atoms, letting the linear coordinate encoder act as a dense autoencoder on the data span. Seeds51/53 expose seven atoms and retain held error0.119/0.111. This is a concrete counterexample to certifying featurizer discovery from reconstruction or compositional test performance alone.

Supplying both membership and gates yields near-exact reconstruction in all seeds. Supplying membership alone leads to nearly all groups active; dictionary directions can rotate within the full reconstructive span, so true index-level membership does not guarantee semantic atom alignment. Supplying gates alone succeeds approximately in one seed (held2.19e-5) and fails in two (0.110–0.118), consistent with earlier membership-search fragility. These controls do not uniquely isolate an optimizer defect from the chosen architecture/objective, but locate failure beyond decoder plumbing.

Exact gate inference over each jointly learned representation only modestly improves its reconstruction-plus-count objective. This contrasts with H693's large gate gap on a correctly structured fixed dictionary. Once joint training co-adapts toward a dense span, replacing its detector alone does not restore the original sparse groups. No universal impossibility is implied: this is one objective, initialization scheme, training budget and noiseless teacher.

Implementation audit: vectorized reconstruction tables match the reference model within3.73e-9on the checked pattern. Exact expected-cost gradients were verified in H695; discrete membership gradients remain a declared surrogate. All source hashes unchanged, finite results, no original-model forwards/new documents, reserved LLM test split untouched. Gate-pattern audit is saved separately alongside full results.

The alternatives are therefore tested implementations, not verified replacements for BSFs. The next bounded experiment should target the dense-span degeneracy before scaling: data-only sparse-direction initialization or a direct active-coordinate cost, with constant-gate/constant-support controls and all seeds. Use teacher structure only for evaluation. In parallel with research prioritization—not a claim of completed deployment—retain the existing BSF manifold/intervention evidence as the current model baseline. A successful synthetic recovery must still pass stability, causal specificity, diverse tasks and matched-budget LLM comparisons. Broad goal remains active.
