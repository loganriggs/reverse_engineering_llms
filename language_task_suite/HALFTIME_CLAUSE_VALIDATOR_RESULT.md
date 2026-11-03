# Conservative halftime sentence bank is too small for a sampled renderer

The previous [opportunity screen](SPORTS_HALFTIME_RULE_OPPORTUNITIES.md) found hundreds of real reports with a numeric pair matching the structured first-half score near halftime wording. Those matches were only lexical opportunities. I added a [complete-sentence semantic screen](audit_halftime_clause_bank.py) that accepts a short clause only when all of these hold: four quarter totals reconstruct the final score; exactly one first-half score pair and no other numeric tokens appear in the sentence; exactly one unambiguous team is named (or the sentence is a neutral tie statement); and the entire normalized sentence matches a limited lead/tie grammar. A named leader must actually lead at halftime, and its score must be the **first** number in the printed pair. Trail/down forms are excluded because the corpus uses both score orders. Extra narrative or numeric claims cause rejection. Source clauses and unhashed forms remain local.

| Source / game split | Near-halftime surface matches | Accepted complete clauses | Distinct accepted forms |
|---|---:|---:|---:|
| RotoWire train | 165 | 0 | 0 |
| RotoWire heldout | 36 | 0 | 0 |
| SBNation train | 761 | 8 | 8 |
| SBNation heldout | 172 | 1 | 1 |

The accepted training examples are four direct lead constructions, one lead-after-time, one lead-noun, one *went into halftime up*, and one neutral tie. Seven adversarial fixtures checked reversed numeric order, false lead/trail/tie claims, extra narratives, and extra numbers; a positive tie fixture also passed. An early version incorrectly accepted a reversed pair in a “led” sentence merely because both numbers matched the box score; the order check fixed it before this result was recorded.

This is a **conservative lower bound**, not a claim that only eight real sentences express halftime state correctly. Many natural reports have embedded clauses, pronouns, score orientation determined by context, or extra claims outside this narrow grammar. Those require a richer typed parser and world model to reslot safely. The current accepted bank cannot supply varied training documents or a credible heldout form panel; resampling eight forms would make the task window repetitive. Keep the hundreds of matched human sentences as an evaluation corpus or source for future semantic annotation, and do **not** treat the broader near-half matches as automatically safe generator templates.

The next broad DGP work should diversify its rules and act process without requiring all rules to co-occur in one document. For halftime-to-final specifically, use a purpose-built typed realization grammar only if its task-bearing spans pass a separate natural-form audit; otherwise this task remains a semantic fixture/control. Model training and weight recovery on a new halftime rule are not justified by this bank.

Aggregate counts and hashed form frequencies are in [the result JSON](results/sports_halftime_clause_bank.json); no source prose is committed.
