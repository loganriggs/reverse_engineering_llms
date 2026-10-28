## H795 — One L2 attention head largely reproduces the source-edge interaction-removal effect

Previous turn was progress: H794 established that replacing the second-source L2 attention output by its additive factorial prediction changes final candidate composition, whereas the early query MLP clamp has negligible effect. H795 reconstructs the specific attention edge and tests all nine heads.

L2H5 here means zero-indexed layer2, head5 (the third transformer block); it is not the earlier L5H5 copying readout. We capture the actual post-RMS-normalization/RoPE queries and keys and the actual mixed local/bus values entering squared attention. For each head h, the first-source-to-second-source contribution is:

    edge_h(a,b) = [(q_h(b) dot k_h(a))/128]
                  * [(q2_h(b) dot k2_h(a))/128] * v_h(a)

The receiver prompt stays fixed. Source1's L1 output is replaced with identity a; source2's with identity b. The signed attention coefficient is not a probability. Project each head's factorial edge interaction through its columns of c_proj, then remove one head's interaction, all edge interactions, or the full module interaction at the second-source output. All nine heads are tested; selection is not based solely on activation norm.

### Exact edge reconstruction

Queries at the second source are exactly independent of the first source's intervention, and first-source keys/values are exactly independent of the second source's intervention at this point. Summing the projected edge interactions reconstructs the full L2 second-source attention factorial interaction with6.43e-7 relative RMS error. Removing all edge interactions reproduces the previous full-module additive-clamp candidate effects within1.05e-5 maximum component error.

This verifies a specific multiplicative source-to-source computation underlying that module interaction. It does not yet tell us whether the head implements lexical equality, a broader similarity, a contextual relation, or some other learned operation.

### Causal head screen

Remaining centered candidate-interaction RMS relative to baseline:

| Format | Donors | Remove all edge-head interactions | Remove only L2H5 interaction |
|---|---|---:|---:|
| Vertical bars | Matching | .5362 | .5351 |
| Colon/newline | Matching | .5340 | .5336 |
| Vertical bars | Mixed | .8913 | .8937 |
| Colon/newline | Mixed | .9291 | .9297 |

Other single-head removals leave ratios close to1 (approximately .996–1.006 across these cells). Their individual effects need not add, and the screen does not rule out conditional roles.

To check more than aggregate similarity, compare the per-example centered candidate effect of removing H5 with the effect of removing all edge interactions. Relative RMS discrepancies are3.53%/3.33% for matching donors and8.75%/5.81% for mixed donors in vertical-bar/colon-newline formats. These are errors in reproducing an intervention effect, not percentages of the entire task explained or unique causal shares.

For matching donors, removing H5 increases mean candidate specificity by .0610/.0587, close to all-head increases .0628/.0594. For mixed donors, the increases are .0263/.0145 versus .0261/.0143. As in H794, the identified interaction partly counteracts the additive donor signal under this hybrid intervention; stronger counterfactual donor promotion is not a native-task improvement claim.

### What has and has not been established

The chain now has both algebraic and causal support: L1 source-output changes alter source1 keys/values and source2 queries; their interaction on the first-to-second-source L2 attention edge reconstructs the full module factorial interaction; L2H5 removal nearly reproduces the all-edge removal effect on final candidate responses. This is a concrete computational relation to investigate, more constrained than a generic claim about overlapping manifolds.

It remains a local result from reused copying examples. We remove only the factorial interaction, not the full head, full edge or all content carried by them. Considerable output non-additivity survives, especially for mixed donors. No equality detector, universal feature dictionary, manifold, minimal circuit, or learned DAG has been established. Head5 was selected after an exhaustive screen on this panel and requires frozen validation elsewhere.

### Verification

6,656 model forwards (512 factor captures plus512factorial states across12 intervention arms). Q/K/V source-independence errors exactly zero. Edge-to-module interaction relative RMS6.43e-7. Capture, no-removal/full-module reference and all-edge versus H794 clamp candidate replays each <=1.05e-5 maximum error. Identity logits <=1.67e-5 across batches. Metrics finite; model/input/reference/runner hashes unchanged. Original reserved test untouched; natural-document pool remains26,112.

Full per-head results and captured factors are saved. Formats share lexical items and the panel is reused for mechanism discovery; apparent consistency across those formats is not an independent head-selection validation.

### Next discriminator

Freeze L2H5 and the first-source-to-second-source edge, then validate against the separate household-value lexical panel previously used for source localization, without reselecting heads. Test full-module versus H5 interaction-removal fidelity, actual donor-specific effects where candidate readouts permit, and whether the relevant Q/K products favor matching versus mismatching identities. Distinguish lexical content effects from generic input similarity before choosing a compact operator or shared-dictionary representation. The edge's exact formula supplies a concrete target for later featurizer fidelity tests; low reconstruction loss alone would not verify it.
