## H629 — Combined early normalized-MLP operators flip the language effect; attention alone does not

The previous turn was progress: H628 validated local-operator propagation and identified blocks6–9 as a useful region for the language “speak” reversal. H629 factors those blocks into the learned residual-skip coefficient, normalized-attention Jacobian, residual addition, normalized-MLP Jacobian, and final residual addition. The current incoming tangent is propagated through each stage. Later blocks and final readout remain native.

Use the same four capital form0 discovery contexts as reference operators. Test attention6/7/8/9 separately, MLP6/7/8/9 separately, all early attention stages, all early MLP stages, and both kinds together. “Normalized MLP” includes the incoming RMS derivative; its internal factors are not separated here. “Attention” includes normalization, query/key/value and output-projection effects, not routing alone. No new feature direction is fitted.

**The microstage factorization is verified.** Its native residual trajectory differs from the whole-block JVP chain by at most2.89e-6 relative norm. The original paired-context early-block replacement reproduces H628 exactly. The native full-logit response still matches measured finite differences within0.000564 relative weighted norm. These checks precede interpretation of replacements.

**Replacing all four normalized-MLP operators flips the “speak” effect on every held-out item.** Native intended-language log-probability derivative is−0.235 on average. All-MLP replacement gives+0.131, whereas all-attention replacement gives−0.193. On the four held-out countries, native derivatives range−0.338 to−0.162; all-MLP replacement yields+0.058 to+0.194. Every individual MLP replacement leaves all four derivatives negative. The combined substitution is sufficient for the local sign change under this reference choice; no single tested substitution is sufficient on these items.

| Replacement in blocks6–9 | Language “speak” response change | Mean intended-token derivative |
|---|---:|---:|
|native|0.0003|-0.2353|
|attn6|0.0982|-0.2172|
|attn7|0.1452|-0.2444|
|attn8|0.1572|-0.2205|
|attn9|0.1310|-0.2107|
|mlp6|0.7130|-0.1847|
|mlp7|0.6244|-0.1130|
|mlp8|0.3653|-0.2381|
|mlp9|0.4775|-0.1094|
|attn_all|0.2317|-0.1930|
|mlp_all|1.6429|0.1308|
|both_factorwise|1.2727|0.0372|
|paired_early|1.2735|0.0370|

All derivatives use addition of the original frozen Attention5 vector as positive. Response change is the native-probability-weighted centered full-logit difference relative to the native tangent norm. Values above1 are not percentages of computation removed. These are local response changes, not finite-amplitude accuracy or ordinary prompt interventions.

MLP7 and MLP9 individually weaken the negative derivative substantially, to−0.113 and−0.109. MLP6 gives−0.185. MLP8 gives−0.238, showing that not every substitution moves the target in the same direction. None of these observations establishes an additive contribution: operator changes affect the incoming tangent at later stages. The joint sign flip could reflect accumulated effects, interactions, or both. An exhaustive subset comparison is needed before naming a minimal combination or claiming synergy.

**The broader panel shows different effects.** Capital derivatives remain positive under all-attention/all-MLP substitution. Language declaratives remain positive but weaken under all-MLP replacement; language question-answer prompts move close to zero. Two plural forms become negative, while the question-answer plural form stays positive. This is not a universal sign switch associated with “MLP replacement.”

| Held-out cell | Native derivative | All attention shared | All normalized MLPs shared | Both factorwise shared | Paired block reference |
|---|---:|---:|---:|---:|---:|
|capital/0|0.5448|0.5876|0.5816|0.6076|0.6067|
|capital/1|2.8760|2.8561|3.2345|3.0780|3.0736|
|capital/2|0.4922|0.4949|0.4706|0.4128|0.4102|
|language/0|0.1148|0.1052|0.0250|0.0302|0.0284|
|language/1|-0.2353|-0.1930|0.1308|0.0372|0.0370|
|language/2|0.0904|0.0932|0.0047|-0.0034|-0.0031|
|plural/0|0.2220|0.2386|-0.0328|-0.0597|-0.0637|
|plural/1|0.3563|0.2360|-0.0141|-0.1505|-0.1535|
|plural/2|0.1178|0.1339|0.1992|0.0664|0.0664|

“Both factorwise” averages attention and MLP Jacobians separately before composing them. “Paired block reference” composes the two native operators within each reference context and then averages whole-block Jacobians. Products and averages need not commute; both are reported rather than conflated. Their language “speak” derivatives happen to be close here (+0.0372/+0.0370). Neither recovers the stronger+0.1308 from MLP-only substitution, so the reference attention operators counteract part of that change in the combined intervention.

Interpretation: the chosen capital-versus-language operator contrast is stronger in the early normalized-MLP sequence than in its attention sequence for the studied sign reversal. This is more specific than the earlier whole-region finding. It does not establish that native attention is unimportant: attention supplies the context, and the tested intervention changes only its first-order operator around fixed native states. Likewise, “MLPs” here includes their normalization and residual interaction; bilinear multiplication itself has not yet been isolated for blocks6–9.

The evidence is consistent with a distributed conditional transformation of a reusable direction. A fixed scalar feature meaning does not capture the evolving response, and an architectural component's role depends on its incoming perturbation and operating state. This remains an adaptive local mechanism study on familiar prompts. Plural derivatives are retained despite corrected competence failures; four held-out country identities are not a broad independent sample.

Next enumerate all16 subsets of the four normalized-MLP replacements, using the same references and frozen direction, to test whether a smaller combination reproduces the joint sign change and how well sums of single-substitution effects predict combinations. In parallel compare sharing bilinear operating context versus sharing input-normalization Jacobians across the early MLP region. That distinguishes accumulated operator effects from interaction and identifies which internal factor carries the context dependence, without selecting a new feature direction.

Audits: cached native contexts from H628 reused with hashes checked;288 native attention-stage evaluations to reconstruct intermediate states;1,440 batched JVP calls; no new whole-model native forwards. Native microfactorization max relative discrepancy2.88e-6; paired reference replay exact; end-to-end finite-difference comparison max0.000563; all results finite. Four reference IDs and fixed item0–3 discovery split retained. FineWeb remains26,112 natural-text documents. Broad research goal active and incomplete.
