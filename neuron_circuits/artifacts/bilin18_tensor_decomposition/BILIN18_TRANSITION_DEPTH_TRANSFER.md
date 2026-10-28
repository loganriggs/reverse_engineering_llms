## H804 — Query response transfer first reverses the gap at L6 among consecutive tested cuts

H803 bracketed a transition between L3 and L8. H804 tests every intervening residual cut L4–L7 with the same response-transfer design, paired zero-delta controls, all-token versus query-only conditions, and fixed local-state pairing. Each replacement retains the recipient baseline and adds the other context's perturbed-minus-baseline response. Layer indices are zero-based.

| Cut | Site | Vertical-bar context gap | Colon/newline context gap |
|---|---|---:|---:|
| No transfer | — | +.3862 | +.2172 |
| L4 | Query | +.3773 | +.2084 |
| L4 | All tokens | -.2435 | +.0294 |
| L5 | Query | +.1257 | +.1069 |
| L5 | All tokens | -.2449 | +.0231 |
| L6 | Query | -.4506 | -.0486 |
| L6 | All tokens | -.4660 | -.0500 |
| L7 | Query | -.4678 | -.1232 |
| L7 | All tokens | -.4739 | -.1173 |

The context gap is different-key minus same-key intended-token log-probability response to the +.50 value perturbation. Query-only swaps have modest mean effects at L4, partial effects at L5, and first reverse the gap at L6 in both formats. L6 transferred gaps have descriptive item-bootstrap intervals [-.6388,-.2875] and [-.0857,-.0160]. L5 gaps remain positive with intervals [.0547,.1914] and [.0547,.1585]. This locates a transition in transferable query-response content; it does not prove that one layer alone computes the relevant function.

All-token transfers already have large effects at L4: the gap reverses in vertical bars and is near zero in colon/newline. This prioritizes information in non-query positions before it becomes strongly transferable at the query. It does not uniquely identify the second source, since every token's residual response is replaced. Later downstream computation and response/background interactions remain part of each measured effect.

Next separate L5/L6 query attention and MLP outputs, transferring each response separately and jointly while retaining paired recipient baselines. The joint module-output arm uses original captured responses, so it tests a controlled replacement rather than the MLP's freely recomputed response after the attention swap. A circuit claim still needs source/head localization and frozen fresh-panel transfer.

3848 model forwards completed; all tracked hashes unchanged. Matched direction mismatch is zero, local clamp relative error 1.40e-8, and every zero-delta full-logit replay is exactly zero error. H800 response reproduction error is at most 5.00e-6. All responses finite. This uses the same 16/10 aligned items and both local-state/query variants; intervals average within item, are descriptive, and have no multiplicity correction. Full-vocabulary accuracy stays 100% in colon/newline and ranges 96.9–98.4% in vertical bars. Hybrid responses are not guaranteed on-manifold. Reserved test untouched; document pool 26,112. No featurizer or manifold newly validated.
