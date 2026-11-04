# Form audit of the corpus-grounded three-rule training mixture

The [joint-model pilot](CORPUS_NONCE_COPY_PILOT.md) trained on a 50% nonce-copy, 25% numbered-list, 25% bracket mix of task-centered 256-BPE-token windows from real FineWeb documents. These are authentic prose windows with small, rule-preserving edits, but task-conditioned sampling can still distort the distribution a model sees. This audit measures that distortion against two fresh, disjoint FineWeb source ranges. It does not certify semantic realism or a full next-token oracle.

For each panel, the [auditor](audit_corpus_three_rule_form.py) samples 400 training windows at the actual 200/100/100 task mix, using one alias draw per distinct nonce source. It also draws 400 random-position 256-token crops from previously unused FineWeb sources with at least 128 tokens. It decodes GPT-2 BPE tokens and compares the first 100 alphabetic words of every window; all 400 windows in each arm have at least 40 such words. No source prose is saved in the aggregate JSONs.

| FineWeb start | Windows | Word types | Singleton types | Top-ten word share | Cross-document repeated-trigram share | First-40 repeated-word share |
|---:|---|---:|---:|---:|---:|---:|
| 350,000 | Training mix | 8,572 | 4,641 | .211 | .041 | .160 |
| 350,000 | Fresh FineWeb | 8,791 | 4,970 | .211 | .036 | .162 |
| 352,000 | Training mix | 8,428 | 4,618 | .210 | .040 | .161 |
| 352,000 | Fresh FineWeb | 8,409 | 4,563 | .216 | .037 | .175 |

This is a substantial surface improvement over the earlier tiny-lexicon toy streams, which had only a few hundred word types and top-ten shares around .37–.43. It is expected because nearly every word here comes from human prose. The mixture's first-100-word lexical statistics are close to the matched FineWeb crops, although these coarse statistics alone do not establish genre, syntax, or dependency realism.

| FineWeb start | Windows | Numbered line | Parenthesis | Colon | Newline |
|---:|---|---:|---:|---:|---:|
| 350,000 | Training mix | .260 | .538 | .463 | .995 |
| 350,000 | Fresh FineWeb | .018 | .455 | .440 | .970 |
| 352,000 | Training mix | .260 | .570 | .460 | .983 |
| 352,000 | Fresh FineWeb | .013 | .432 | .405 | .988 |

The large numbered-line gap is imposed by conditioning 25% of training windows on the list task: every selected list window contains such a line. Bracket-conditioned windows have a parenthesis rate around .78–.80, raising that cue above the reference. These are window-level rates, not estimates of natural task prevalence. The comparison also differs in crop selection: task windows are centered to retain their source and answer, whereas reference windows use random positions within source documents. That is part of the actual training-distribution shift and must not be explained away as if it were a measurement error.

The [source-overlap audit](results/corpus_nonce_list_source_overlap.json) found only 18 strict nonce/list intersections in 100,000 training FineWeb sources and one in 10,000 validation sources. Their test ranges do not overlap. Simply intersecting the current selected caches will not yield a credible three-rule crossed-document test.

The next DGP should separate two distributions. A **reference stream** samples whole source documents without conditioning on a task, applies supported local transformations when a rule is eligible, and chunks the resulting text in ordinary document order. It should report how often each rule is actually present and use independent source splits. A **diagnostic or enriched stream** may condition on rare rule sites, use balanced edit arms, and upweight their losses so small models can learn them; its task and cue rates must be reported separately. Whole-document and answer-window form, visible-prefix rule checks, source-level duplicate screens, task-conditioned model competence, and held-out causal edits remain separate gates. New, more common language dependencies are preferable to forcing three current rare sites into one document with conspicuous inserted questions.

The first FineWeb streaming run wrote its complete aggregate, then aborted in native library cleanup. The second panel completed with exit code 0 after following the existing explicit-exit convention for this streaming stack. Both JSON results parse and contain only statistics and source-offset metadata: [panel 350,000](results/corpus_three_rule_form_vs_fineweb_start350000_n400.json), [panel 352,000](results/corpus_three_rule_form_vs_fineweb_start352000_n400.json).
