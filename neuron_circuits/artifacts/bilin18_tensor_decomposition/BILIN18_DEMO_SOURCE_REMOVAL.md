## H677 — Query content matters in competent tasks; head4 demonstration-answer writes do not explain it

H676 fixed five competent family/format cells before these interventions. H677 runs every one of192 prompts, not only correct examples. It removes the final-position layer14 attention contribution from (1) query-input tokens through head4, (2) query-input tokens through all9heads, or (3) all four demonstration-answer tokens through head4. Native baseline is replayed. Query and answer spans are identified by tokenizer offsets; multi-token query inputs are included completely.

The primary analysis retains all eight items in each prequalified cell, under task and copy instructions. Failed competence cells remain in the complete results and secondary table. These edits remove specific writes, not source representations; the rest of the model runs normally. Interventions differ in token counts and vector norms, so this is not an equal-strength comparison of abstract importance.

### Prequalified cells

Each entry below gives correct-target CE change (positive=worse), followed by full-vocabulary top1 correct/8. Native KL and per-example results are saved in JSON. The target is the actual instruction's answer, including copying targets.

| Family | Format | Context | Native correct/8 | Query head4 ΔCE; correct | Query all ΔCE; correct | Demo answers head4 ΔCE; correct |
|---|---|---|---:|---|---|---|
|plural|arrow|task|7|+0.1366; 7|+0.1165; 6|-0.0469; 8|
|plural|arrow|copy|7|+0.1136; 7|+0.1496; 7|-0.0455; 7|
|plural|labeled|task|6|+0.1750; 6|+0.3177; 7|-0.0153; 6|
|plural|labeled|copy|8|+0.1019; 8|+0.2861; 8|-0.0039; 8|
|successor|arrow|task|7|+0.2374; 7|+0.2634; 7|-0.1387; 7|
|successor|arrow|copy|8|+0.0014; 8|+0.0180; 8|-0.0080; 8|
|agreement|arrow|task|8|-0.0002; 8|-0.0001; 8|-0.0000; 8|
|agreement|arrow|copy|8|-0.0006; 8|-0.0005; 8|-0.0000; 8|
|agreement|labeled|task|8|-0.0111; 8|-0.0198; 8|-0.0000; 8|
|agreement|labeled|copy|8|-0.0015; 8|+0.0001; 8|+0.0000; 8|

Query-input removal worsens mean task CE for both plural formats and month succession. Month head4 removal increases CE by0.2374 while accuracy remains7/8; removing all heads adds only modest further loss (0.2634). Agreement is largely unaffected or slightly improved by these particular removals. Thus this local mechanism contributes to some successful transformations but is not a common necessary operation for every competent task.

In labeled plurals, all-head removal raises accuracy6/8→7/8 while worsening mean CE by0.3177. This is a concrete accuracy/loss disagreement, not contradictory evidence: probabilities on other examples can worsen while one crosses the top1 boundary.

Removing demonstration-answer writes through head4 does not worsen mean CE in any qualifying task cell. It modestly improves month CE (−0.1387) and arrow plural CE (−0.0469; accuracy7/8→8/8). This disfavors direct demonstration-answer support through this specific head at this position. It does not show demonstrations are irrelevant: other heads, earlier positions or earlier layers may have used them to establish the operation.

### Native head4 query coefficients

| Family | Format | Task mean sum | Copy mean sum |
|---|---|---:|---:|
|plural|arrow|-0.10111|-0.12946|
|plural|labeled|-0.23079|-0.26940|
|successor|arrow|0.34829|0.00004|
|agreement|arrow|-0.04401|-0.10011|
|agreement|labeled|-0.11168|-0.13211|

Single-token plural/month queries make these directly interpretable coefficients. Agreement sums multiple token coefficients and should not be treated as one gate controlling a single value. Month task mean is+0.3483 versus approximately0 in copying; plural coefficients remain negative under both instructions. These competent prompts retain task-dependent gating but do not all implement transformation through the same gate sign.

The next causal test is paired gate transplantation on the single-token qualified cells: keep the receiver value and all other computation intact while replacing only its query-head4 coefficient with that from the matched opposite instruction. This distinguishes a causally relevant instruction gate from an associated write. Analyze both directions, all eight items, plural formats and month arrows. Success would establish a local gate effect, not sufficiency for switching the entire task.

### Secondary task cells that failed the fixed competence threshold

| Family | Format | Native correct/8 | Query head4 ΔCE | Query all ΔCE | Demo head4 ΔCE |
|---|---|---:|---:|---:|---:|
|antonym|arrow|5|+0.0024|+0.0691|-0.0294|
|antonym|labeled|1|-0.0107|+0.0862|-0.0117|
|capital|arrow|4|-0.5085|-0.5590|+0.0459|
|capital|labeled|2|-0.4646|-0.4903|-0.1599|
|successor|labeled|5|+0.2944|+0.3229|-0.0545|
|addition|arrow|2|+0.0534|+0.0441|-0.0150|
|addition|labeled|2|+0.0254|+0.0357|-0.0216|

All copy-cell results, removal norms and KL values are retained in the result JSON. No inference of universal source importance or absence of redundancy follows from this one layer/head comparison.

Audits:768 whole-model forwards (192×4 conditions). Native logits replay exactly. Full attention contribution reconstruction maximum relative error5.204e-7. Query spans match declared input strings; four nonempty demonstration-answer tokens per prompt and no query/answer overlap asserted. All tracked source hashes unchanged and logits finite. Role maps, removal vectors, coefficients and output logits retained. FineWeb remains26,112 unique documents. Prior turn was experimental progress. Broad goal remains active and incomplete.
