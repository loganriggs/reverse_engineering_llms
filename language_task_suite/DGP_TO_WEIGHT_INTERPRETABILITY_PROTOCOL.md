# DGP variation as a test of weight-based interpretability

The main outcome is whether a method applied to **weights** recovers rules and reusable structure that a bilinear transformer actually learned from a known, variable, language-like data-generating process (DGP). Capability is a prerequisite for a particular rule's weight-level evaluation, not the target to maximize. The large probabilistic-grammar proposal is a source of components, not a requirement to build a perfect language oracle before training.

## Experimental unit

Sample complete documents from one joint DGP. Local statistics (bigram, skip trigram), copying/induction, bracket and quotation closure, entity reference, agreement, arithmetic, genre, and layout may coexist and interfere in a document. Record the sampled latent events and the rule responsible for each prediction site. Keep these labels separate from anything used to fit a supposedly unsupervised weight decomposition.

Vary DGP knobs with controlled seeds: lexical frequency tail and document burstiness; paraphrase and syntactic realization entropy; genre and discourse transitions; each rule's prevalence and context distance; entity/distractor count; and whether rules are independent, correlated, or condition one another. Vary model depth/width/training seed separately. Start with a small factorial covering lexical/surface variation, rule prevalence, and cross-rule coupling while holding total tokens and architecture fixed; extend only after the first comparison reveals which factor matters. Compare generated samples and independent FineWeb panels on word and n-gram frequencies, vocabulary growth, repetition by distance, sentence/paragraph length, layout, and qualitative coherence. Report gaps as continuous measurements, not a single pass/fail score.

## Gate each interpretation claim on learned behavior

For each rule family, use unseen generated documents, balanced alternatives, supported counterfactual edits, and held-out combinations. A rule counts as learned only if the model predicts its target beyond local/frequency shortcuts and responds in the specified direction when the relevant cause changes. Report uncertainty over documents and model seeds. If a rule is absent or weak, record that as a learning result; do not score the absence of a weight circuit as an interpretability failure. Also preserve untrained and shuffled-label controls.

An exact marginal next-token oracle helps quantify regret and contextual information when available, and small exact fixtures remain useful correctness tests. It is not needed to label sampled rule sites, run controlled edits, or compare weight methods. Do not substitute the generator's hidden sampled state for the visible-prefix predictive target when making oracle claims.

## Weight-method comparison

On each behaviorally learned rule, compare the original iterative L1/prune method with final **no-L1** fine-tuning; direct embedding-to-unembedding and QK/OV path inspection; weight-only canonical/bond bases and gauge/sparse rotations; and task-conditioned or data-weighted variants as explicitly separate methods. Match retained function fidelity, total nonzeros/edges, and analysis budget where possible. Score whether a proposed component is causally necessary/sufficient for the edit, specific to the rule rather than a correlated token class, stable across seeds and surface variants, and reusable on held-out rule combinations. Reconstruction or visual sparsity alone is not evidence of rule recovery. A task-conditioned method can use rule labels for fitting, but must be evaluated on distinct documents, edits, and rule combinations; label-free discovery uses labels only after fitting.

## Immediate next comparison

Build a coherent generator revision from the existing sampled-rule machinery with (1) at least two surface/lexical-variation settings and (2) independent versus coupled rule settings. Keep explicit per-site provenance and supported edit pairs. Train a small fixed-budget depth-1/depth-2 pilot across these settings, score learned-rule panels, then compare a small set of weight methods on the same learned rule. Include the older V3 sampled generator as a controlled, less realistic baseline. Do not resume PCFG-only scale sweeps until a specific weight-recovery question needs them.
