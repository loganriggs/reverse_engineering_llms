"""Reproduce a hand-inspected target-site ledger from a local FineWeb preview.

Stores source hashes and offsets, never source passages. Offsets are character
offsets in the first 1,200-character preview, not tokenizer positions.
"""

import argparse
import json
from pathlib import Path


# Each entry is a concrete next-text site, not a claim that the target is
# uniquely implied. Occurrence selects repeated strings in the preview.
SITES = [
    dict(row=23006, term='She', occurrence=0, family='coreference',
         evidence='young female witness introduced and acted in prior clause',
         judgment='probability_shift', alternative='grammar and nearby female noun suffice',
         edit='change witness gender while preserving the attack event'),
    dict(row=23014, term='third floor', occurrence=2, family='procedure',
         evidence='goal names third floor; earlier steps go first then second',
         judgment='visible_but_leaked', alternative='copy explicit goal or repeat floor phrase',
         edit='replace goal and step sequence independently'),
    dict(row=23019, term='11', occurrence=1, family='scoped_record',
         evidence='second ship record begins after first record with same room number',
         judgment='exogenous_record_value', alternative='first record category is 11B',
         edit='swap ship-specific categories while preserving room number'),
    dict(row=23033, term='5-4', occurrence=1, family='score_update',
         evidence='body reports 4-2 then three runs for trailing team',
         judgment='visible_but_leaked', alternative='headline already states 5-4',
         edit='mask headline score; edit either starting score or run count'),
    dict(row=23037, term='Mathis', occurrence=1, family='speaker_attribution',
         evidence='commission chair introduced by full name before quote',
         judgment='visible_identity', alternative='local quote attribution pattern',
         edit='swap chair identity while preserving quote'),
    dict(row=23038, term='earlier', occurrence=0, family='contrastive_schema',
         evidence='daylight-saving start says later; current row says ended',
         judgment='probability_shift', alternative='common daylight-saving convention',
         edit='swap start/end row labels and directions'),
    dict(row=23043, term='25mm', occurrence=0, family='product_schema',
         evidence='product specifications contain case and strap measurement fields',
         judgment='exogenous_value', alternative='unit and field type predictable, value is not',
         edit='change case dimension independently of strap dimension'),
    dict(row=23062, term='Madison', occurrence=0, family='entity_naming',
         evidence='puppy later called dog, followed by a proper name',
         judgment='exogenous_name', alternative='name is introduced at this site, not recoverable',
         edit='change name and test only later references'),
]


def build(preview_path):
    rows = {row['index']: row for row in json.loads(Path(preview_path).read_text())}
    out = []
    for site in SITES:
        source = rows[site['row']]
        text = source['preview']
        starts = []
        at = 0
        while (at := text.find(site['term'], at)) >= 0:
            starts.append(at)
            at += len(site['term'])
        if len(starts) <= site['occurrence']:
            raise ValueError(f"missing target {site['row']} {site['term']}")
        start = starts[site['occurrence']]
        out.append({
            'source_row': site['row'], 'source_sha256': source['sha256'],
            'target': site['term'], 'start_char': start,
            'end_char': start + len(site['term']),
            'family': site['family'], 'evidence_paraphrase': site['evidence'],
            'predictability': site['judgment'],
            'competing_explanation': site['alternative'],
            'disambiguating_edit': site['edit'],
        })
    return {'source': 'FineWeb sample-10BT first-1200-character previews',
            'scope': 'eight purposively chosen sites; no prevalence estimate',
            'sites': out}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-json', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build(args.preview_json)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(f"wrote {len(result['sites'])} verified target offsets")
