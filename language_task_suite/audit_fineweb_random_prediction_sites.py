"""Reproduce a one-reader random FineWeb prediction-site annotation.

Reads the temporary 64-document preview produced by
audit_fineweb_predictive_inventory.py. Stores hashes, word targets, and
paraphrased annotations, not document passages. Selection is independent of
the manual labels below.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


SEED = 24011
EXCLUDE = {23006, 23010, 23014, 23019, 23033, 23038, 23043, 23062}
TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

# Primary mechanism is the best *available* explanation, not a recovered
# fact about the author or a claim that the selected word was necessary.
LABELS = {
    23011: ('document_context', 'probability_shift', 'local',
            'Court testimony is being discussed; a deictic location word is plausible.',
            'there / today', 'replace the testimony location; natural edit is hard'),
    23028: ('semantic_frame', 'probability_shift', 'local',
            'An injured cricketer is describing what happened to his shoulder.',
            'hurt / injured', 'change the injury description'),
    23022: ('collocation', 'probability_shift', 'local',
            'A player evaluation uses the familiar phrase about breaking down on a play.',
            'in / the', 'replace the sports phrase with a matched alternative'),
    23005: ('page_schema', 'probability_shift', 'document',
            'An instructional course page moves from benefits to course content.',
            'Topics / Skills', 'change the page section plan'),
    23049: ('collocation', 'probability_shift', 'local',
            'Government spending is followed by the conventional phrase about a fiscal hawk.',
            'budget / political', 'change the preceding spending frame'),
    23004: ('quote_intent', 'probability_shift', 'local',
            'A first-person testimonial starts a positive sentence about classroom work.',
            'enjoy / like', 'change the quoted speaker sentiment'),
    23048: ('markup_schema', 'probability_shift', 'local',
            'Scraped music text contains an embedded artist-information tag.',
            'another markup tag or plain text', 'remove or change markup template'),
    23045: ('enumeration', 'probability_shift', 'document',
            'A doctrinal explanation proceeds through named light properties in sequence.',
            'Then / Now', 'change the discourse ordering'),
    23053: ('page_schema', 'probability_shift', 'document',
            'A careers page shifts from download links to a tool-use instruction.',
            'the / our', 'change whether a tool has been introduced'),
    23020: ('wordplay_reuse', 'visible_reuse', 'both',
            'A humorous passage repeatedly uses the turkey-dog combination.',
            'Pilgrim / hot', 'replace the repeated animal/food motif'),
    23001: ('collocation', 'probability_shift', 'local',
            'A services page has a heading of the form Subject Matter expert.',
            'area / topic', 'change the professional heading'),
    23042: ('genre_claim', 'probability_shift', 'document',
            'A promotional architecture article describes a novel local home project.',
            'new / fully', 'change the novelty claim'),
    23032: ('new_named_entity', 'new_fact', 'document',
            'An event preview introduces a cathedral as one tour location.',
            'another church name', 'change the venue identity'),
    23029: ('new_named_entity', 'new_fact', 'document',
            'A travel account lists a Canadian Rockies itinerary; this stop is newly supplied.',
            'Banff / another town', 'change the itinerary stop'),
    23055: ('syntax_semantics', 'probability_shift', 'local',
            'A route description continues after wandering up a backstreet.',
            'through / by', 'change the route relation'),
    23063: ('quote_intent', 'probability_shift', 'local',
            'A quoted first-person remark follows an account of answering questions.',
            'I / That', 'change the speaker wording'),
    23018: ('new_named_entity', 'new_fact', 'document',
            'A nostalgic car fantasy introduces a particular vehicle model.',
            'another car make', 'change the vehicle identity'),
    23044: ('honorific_style', 'probability_shift', 'local',
            'A church-history page introduces the designer as a named parishioner.',
            'a first name without title', 'change the biographical style'),
    23008: ('phrase_reuse', 'visible_reuse', 'both',
            'A resolution essay repeats a phrase about the new year and yearly change.',
            'month / season', 'replace the repeated time frame'),
    23061: ('legal_register', 'probability_shift', 'local',
            'An allegation is followed by quoted legal-style harm wording.',
            'severe / significant', 'change the quoted legal wording'),
}


def build(preview_path):
    rows = json.loads(Path(preview_path).read_text())
    assert len(rows) == 64
    assert [r['index'] for r in rows] == list(range(23000, 23064))
    rng = random.Random(SEED)
    eligible = [r for r in rows if len(r['preview']) >= 900
                and r['index'] not in EXCLUDE]
    selected = rng.sample(eligible, 20)
    assert {r['index'] for r in selected} == set(LABELS)
    sites = []
    for row in selected:
        words = [m for m in TOKEN.finditer(row['preview'])
                 if 300 <= m.start() < 900]
        match = rng.choice(words)
        mechanism, predictability, scope, evidence, alternatives, edit = LABELS[row['index']]
        sites.append(dict(source_row=row['index'], source_sha256=row['sha256'],
                          target=match.group(), start_char=match.start(),
                          end_char=match.end(), mechanism=mechanism,
                          predictability=predictability, evidence_scope=scope,
                          evidence_paraphrase=evidence,
                          plausible_alternatives=alternatives,
                          proposed_edit=edit))
    return dict(source='HuggingFaceFW/fineweb sample-10BT',
                selection='20 distinct eligible documents sampled from offsets '
                          '23000..23063 after excluding 8 purposive examples; '
                          'uniform random word start in characters 300..899; seed 24011',
                caveat='One annotated site per document, one reader, short previews; '
                       'not a token- or corpus-weighted prevalence estimate. '
                       'Labels are hypotheses about useful predictive evidence.',
                n_eligible_documents=len(eligible), n_sites=len(sites),
                predictability_counts=dict(Counter(s['predictability'] for s in sites)),
                mechanism_counts=dict(Counter(s['mechanism'] for s in sites)),
                sites=sites)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.preview_json)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['predictability_counts'], result['mechanism_counts'])
