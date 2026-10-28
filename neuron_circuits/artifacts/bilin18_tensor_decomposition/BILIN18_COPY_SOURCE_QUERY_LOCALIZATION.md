## H790 — Value-source swaps reveal stronger L1 effects and a non-additive two-source interaction

Previous turn was progress: H789 failed to clearly replicate the final-position L4 copying lead across formats. H790 tests the prespecified source/query localization alternative using native MLP-output replacements, rather than scaling synthetic directions.

On the H789 copying panel, replace the donor MLP output at the first value occurrence, second value occurrence, both values, final query token, or both values plus query. Layers1/4, all sites, reciprocal task edits, answer-preserving key renames and identity controls were specified before outcomes. Token offsets identify the two single-token value occurrences; exact token IDs are checked. Paired runs require equal sequence length and matching source/query positions. Downstream computation is fully live. This is a whole-model intervention readout, distinct from earlier isolated L5H5 value-path tests.

### Main results

Mean donor-directed answer-margin changes for task edits:

| Format | Layer | First source | Second source | Both sources | Query | Sources + query |
|---|---:|---:|---:|---:|---:|---:|
| Vertical bars | 1 | +1.332 | +1.310 | +.958 | -.021 | +.935 |
| Colon/newline | 1 | +1.523 | +1.380 | +.745 | -.011 | +.736 |
| Vertical bars | 4 | +.118 | +.051 | +.177 | -.017 | +.166 |
| Colon/newline | 4 | +.065 | +.031 | +.095 | +.008 | +.096 |

The L1 single-source effects are consistently donor-aligned in both formats. Descriptive item-bootstrap intervals for first/second source are [1.087,1.579]/[1.033,1.580] in vertical bars and [1.226,1.841]/[1.152,1.637] in colon/newline. Each cell has16 lexical items,32 reciprocal task edits. Both formats have strong native task competence. These observations support location-specific sensitivity and show that weak query-position interventions did not imply weak source-position effects.

However, native source-output changes have larger norm than query changes. This is not a matched-energy site ranking or proof that every source direction is more causally important. L4 source changes have larger norms than L1 source changes but weaker answer effects, so input/output vector norm alone is also insufficient to explain the ranking.

No tested task intervention switches the binary answer preference to the donor (0/32 per site/layer/format). The interventions move margins but do not transplant the complete task computation. This is a donor-output substitution experiment, not proof of necessity or a minimal circuit.

### The informative non-additivity

At L1, replacing both value sources has less mean effect than replacing either individually, despite changing more coordinates. The RMS discrepancy between joint-source effect and the sum of separate-source effects is1.980 for vertical bars and2.473 for colon/newline. At L4 those discrepancies are only .0179/.0078. Adding the query to both-source swaps changes relatively little at either layer.

This sharpens the research question. A single-source swap creates a hybrid disagreement between the repeated value occurrences; swapping both makes the MLP outputs mutually consistent, although other residual paths remain receiver-native. The stronger single-source margin shift could therefore reflect disrupted agreement or suppression of the receiver answer, rather than greater transfer of donor identity. H790 records answer-margin differences and cannot distinguish donor promotion from receiver suppression. The non-additivity is observed; that proposed explanation remains a hypothesis.

### Alignment, controls and numerical audit

All64 reciprocal task edits and32 identity edits survive alignment. Only7/32 nuisance edits survive (1 vertical-bar case,6 colon/newline cases);25 key-renaming edits are excluded by the predeclared token-length/role-position rule. These sparse nuisance results cannot establish broad context invariance. All exclusions and token-role manifests are saved. The final query token is the last key token in vertical-bar prompts but a colon in colon/newline prompts, another reason not to equate the two query interventions semantically.

The initial absolute query-activation replay threshold of1e-4 failed. A diagnostic rerun localized the discrepancy to native MLP activations: maximum component error .001831 after different batching, while native logits differed by2.58e-5 and reference query-swap margins by5.73e-6. The final audited run additionally measured relative activation error (<=1.25e-6 L2), retained the1e-4 behavioral tolerance, and verified exact requested replacements. Identity logits differed by<=2.96e-5. This supports numerical replay agreement at activation scale, rather than silently treating the absolute threshold as passed.

Final audited run:1,126 model forwards (96 native contexts plus1,030 interventions). Two completed diagnostic attempts preceded it, each also1,126 forwards; total executed this experiment3,378. No failed-run results are used for scientific inference. All final metrics finite; source/model/input hashes unchanged. Original reserved test untouched; natural-document pool remains26,112. Bootstrap intervals cluster reciprocal edits by lexical item, are descriptive, and have no multiplicity correction. Formats share lexical items. This reused panel is localization discovery, not an independent confirmation set for a selected site.

### Decision and next discriminator

Prioritize native L1 value-source transformations over the weak final-position feature lead, but do not label the source effects a verified semantic manifold. The immediate experiment should distinguish donor promotion, receiver suppression and repeated-value inconsistency: measure both candidate log-probabilities separately and the full candidate distribution, retain single/both-source conditions, and include unrelated donor identities. Lock the site choices and use fresh prompts for confirmation. A reusable feature should transfer the intended identity specifically, rather than merely reduce confidence when copies disagree. If specificity transfers, compare compact shared representations against full source-output replacements; if not, pursue consistency computation rather than a dictionary-of-identities account.
