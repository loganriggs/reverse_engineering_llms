# Selected historical excerpts (not a new endorsement)


## FINDINGS.md, original lines 350–383

350:   easier with stations named (2.5/3.8 matchers).
351: - **E. FINDINGS hygiene:** keep this file ≤ 2 pages; add every closed arc as it lands.
352: 
353: ## 14. mlp0 is a current-token table plus an attn0-delivered correction — and the measurement that shows it took three tries (§1659–§1661)
354: 
355: **The finding.** In the running model, mlp0's per-token lookup table recovers **90.27%**
356: of its 0.855-nat mean-ablation stake on positions the table covers. Freezing attn0 at its
357: optimal constant raises that to exactly **100%**. So mlp0's un-tableable residue — about
358: 0.083 nats, the quantity `_mlp0_dossier` listed as open — is **entirely context that attn0
359: wrote into the stream**. mlp0 itself computes no context-dependence. It is a token table
360: plus a correction it inherits.
361: 
362: **Correction (§1662), applied here rather than left in the ledger.** The sentence above
363: originally read that *100% of the residue is attn0's write*. That is architecture, not a
364: finding: attention is the only module in a transformer that moves information between
365: positions, so an MLP with its attention inputs frozen is token-deterministic by
366: construction and its covered table is exact necessarily. The frozen arm is the instrument
367: check; the empirical content is the live 90.27%. **Which** attention delivers the residue
368: is a real question and is not answered by this run.
369: 
370: **The front band, same protocol (§1662).** Live covered ceilings are not ordered by depth:
371: mlp0 90.27%, mlp1 96.01%, mlp2 76.98%, mlp3 67.55%. mlp1 is the anomaly twice -- the most
372: tableable of the four despite sitting above mlp0, and carrying a 7.005-nat stake against
373: 0.86/0.77/0.62 for its neighbours. Its 3.99% residue is therefore 0.279 nats, the largest
374: absolute un-tableable quantity in the band. The instrument check passes at all four sites.
375: 
376: **The instrument check, and why it matters more than the number.** With attn0 held
377: constant, the residual stream below mlp0 is embedding + constant, and MLPs are
378: position-wise — so mlp0 is a deterministic function of the current token and a covered
379: table must reproduce it exactly. The frozen ceiling therefore has a value known BEFORE the
380: run: 1.0. Observed: 1.0, with `ce_table = ce_live = 3.50924` equal to five decimals.
381: 
382: Two earlier versions of this measurement failed that check and **neither was reported**:
383: 

## FINDINGS.md, original lines 411–452

411: ablation. Eighteen least-squares linear maps of the residual stream, fitted in stack order,
412: reproduce **60.81%** of that. This is the first account in this project covering every MLP
413: in the model rather than a hand-picked site.
414: 
415: **The model has three regimes, and they are qualitatively different.** Measured as joint
416: substitutions per band, one protocol, with a known-answer instrument check reading 100.00%
417: at every band:
418: 
419: | band | stake | token table | linear map | quadratic remainder |
420: |---|---|---|---|---|
421: | front mlp0–3 | 4.39 | **76.45%** | 68.68% | 31.32% |
422: | middle mlp4–15 | 2.65 | 21.73% | **62.33%** | 37.67% |
423: | late mlp16–17 | 0.72 | 51.02% | **83.60%** | 16.40% |
424: 
425: The front band is the only place in bilin18 where a lookup beats a computation — a token
426: table with access to nothing but the token beats a linear map with access to the whole
427: residual stream. The middle band is where the model stops being a lookup. Since every MLP
428: here is purely bilinear, `1 − (linear ceiling)` prices what the quadratic form does that no
429: linear function can imitate: **bilinearity peaks in the middle and largely switches off at
430: the top**, rather than accumulating with depth.
431: 
432: At per-site grain the band verdict sharpens: **mlp0 and mlp1 want a table, mlp2 is
433: indifferent, mlp3 does not** (whole-stack program: 54.28% with no tables, 57.25% at mlp0–1,
434: 57.29% at mlp0–2, 56.29% at mlp0–3).
435: 
436: **Two structural results about the model.** The twelve middle MLPs are individually below
437: the instrument's floor (0.026–0.106 nats) but carry a **2.645-nat joint stake** — 4.26× their
438: sum. They were never small; they are redundant, and single-module ablation cannot see them.
439: The same shape appeared inside mlp1: freezing attn0 removed *none* of its un-tableable
440: residue, freezing attn1 removed 19%, and the best of eighteen heads reached 12.68% against a
441: 5.56% uniform baseline. Redundant, not sparse — twice, at two grains.
442: 
443: **Five methodological results, each of which cost a wrong answer first.**
444: 
445: 1. **Position-wise masking at scoring time cannot isolate a position-wise intervention**
446:    (LESSONS 27). Excluding uncovered positions from a CE average leaves the substituted
447:    forward pass intact, and attention mixes the error into the positions you kept. On a
448:    quantity whose true value was derivable in advance (1.0): substitute-everywhere +
449:    mask-score gave 0.5583; substitute-only-where-valid gave 1.0000.
450: 2. **Joint substitution compounds off-distribution** (LESSONS 28). Maps fitted against the
451:    real model and installed everywhere: 12 sites → 62.33%, 18 sites → **−42.99%**, worse
452:    than constant ablation. Compiling bottom-up fixes it completely (54.28%) while moving a
