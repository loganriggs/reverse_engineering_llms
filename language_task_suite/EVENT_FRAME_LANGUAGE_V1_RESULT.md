# Two realizations of one sampled event: correct oracle, failed form

The [fresh V14 form audit](V14_FRESH_FRAME_BOTTLENECK.md) showed that the next language generator needs event-level variation rather than another fixed tail phrase. [`event_frame_language_v1.py`](event_frame_language_v1.py) is a **small correctness fixture** for that architecture. It represents an event as adjective, object, verb, destination, and person, then realizes the same event in either a topical clause (“As for the …”) or a passive clause (“The … was …”). Both clauses put the adjective and object before the verb, so local adjective/object proposals, prior-object verb induction, and prior-place skip proposals compete at the ordinary verb slot in both forms. The object slot likewise mixes a lexicon prior, adjective bigram, prior-adjective object induction, and prior-place skip. The lexicon priors use a configurable truncated Zipf exponent and give every candidate positive support.

The complete event law is the same in both forms. Conditional on visible previous events (H), it factors as

\[
p(a,o,v,l,p\mid H)=p(a)p(o\mid a,H)p(v\mid a,o,H)p(l)p(p).
\]

The topical order samples person before verb; the passive order samples person after place. Because person is independent of the other event fields, multiplying these factors in either order gives the same event distribution. The first token (`As` or `The`) exposes the frame, so the exact next-token oracle needs no private frame label once the first token is visible. At the start it returns the frame mixture. This is a deliberate first case; a future grammar with ambiguous first tokens must marginalize frame alternatives from the visible prefix.

The [audit](audit_event_frame_language_v1.py) replayed every token in two independent 128×256-token samples, checking normalization, vocabulary support, positive sampled-token probability, and agreement with the sampler's returned distribution. It parsed 2,681/2,669 complete events and observed 2,772/2,761 mixed-object and 2,737/2,732 mixed-verb sites. Among those 5,509/5,493 rule sites, 3,487/3,398 had at least two **non-lexicon** proposals supporting the same output token. A held event had equal object and verb distributions in both frames; the two complete realizations had the predicted probability ratio. Supported edits changed the local adjective proposal and moved exactly .15 verb-induction mass or .20 object-induction mass when a prior event was edited. These checks validate the intended visible-prefix rule algebra, not model learning.

The form screen is decisively negative. On the same first-100-word metric and FineWeb offset used by the V14 audit:

| 128 windows | FineWeb | V14, two seeds | Event-frame V1, two seeds |
|---|---:|---:|---:|
| Distinct word types | 4,047 | 326 / 332 | 39 / 39 |
| Distinct sentence starts | 528 | 59 / 60 | **2 / 2** |
| Top-ten word share | .209 | .365 / .369 | .628 / .633 |
| Cross-document repeated-trigram share | .016 | .634 / .632 | **.988 / .988** |

The new grammar is **less** like real language than V14. It also omits V14's source, bracket, attribute, price, and document-layout families. It is not an acceptable replacement DGP and should not be used to claim improved interpretability or train the main model. Its value is the tested separation between a semantic event, its competing rule proposals, and its realization order.

**Next implementation stage:** expand the event/world lexicon and add several syntactically different, variable-length clause families under a document-level act/genre process. Keep exact visible-prefix replay and frame-equivalence checks, then reintroduce bracket and stateful source/attribute tasks as event interactions rather than separate answer-only blocks. The gate is improvement in *both* complete-document and task-window form on disjoint FineWeb samples while preserving rule-site density and grammar-valid edits. Only then train small bilinear models and compare weight-based recovery, sparsity, and canonicalization on the rules that pass behavioral and causal checks.

The [aggregate JSON](results/event_frame_language_v1_n128.json) contains counts and form statistics, no source text.
