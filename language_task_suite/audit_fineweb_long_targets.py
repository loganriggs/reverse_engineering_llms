"""Pin ten manually selected target spans in randomly selected long pages.

The document sample was fixed before choosing the targets; target selection
is purposive within each document. No source prose is exported.
"""

import argparse
import json
import random
from pathlib import Path


SEED = 25017
SITES = {
    25002: dict(target='Mark', start=1239, alternative='James',
                mechanism='scoped_biography', predictability='visible_identity',
                evidence='Current biography introduced Mark Worthington; other biographies name different attorneys.',
                shortcut='Same first name appears earlier in the current biography teaser.'),
    25020: dict(target='Bronx', start=2144, alternative='Brooklyn',
                mechanism='document_topic', predictability='probability_shift',
                evidence='The borough and local community recur throughout the interview.',
                shortcut='Many recent repeats make this an easy topic-copy site.'),
    25039: dict(target='Bryant', start=3864, alternative='Rogers',
                mechanism='entity_role', predictability='visible_identity',
                evidence='The article distinguishes several prospects, including Bryant and Rogers, by their situations.',
                shortcut='The continuation can use topical association, not necessarily a structured entity graph.'),
    25029: dict(target='Wood', start=2678, alternative='Kapler',
                mechanism='entity_role', predictability='visible_identity',
                evidence='An earlier player is the personalized-greeting example; a later sentence names a donating player.',
                shortcut='The donation fact is introduced at this point, so exact donor choice is not compelled.'),
    25026: dict(target='Ingram', start=2876, alternative='Tebow',
                mechanism='speaker_attribution', predictability='probability_shift',
                evidence='The article focuses on Ingram before a long table of opponents and rankings.',
                shortcut='Topic prominence alone may explain the speaker name.'),
    25009: dict(target='Malygos', start=2537, alternative='Kalecgos',
                mechanism='entity_role', predictability='probability_shift',
                evidence='The page earlier distinguishes a former leader from the current leader of the blue dragonflight.',
                shortcut='The target begins a newly supplied lore claim; prior role does not determine wording.'),
    25006: dict(target='Rosebel', start=1578, alternative='Armina',
                mechanism='technical_topic', predictability='probability_shift',
                evidence='The article names both formations; recent mineralization discussion concerns the Rosebel deposit.',
                shortcut='A nearby Rosebel mention suffices; long context is unnecessary.'),
    25001: dict(target='3.', start=2134, alternative='4.',
                mechanism='numbered_schema', predictability='visible_sequence',
                evidence='An interview presents numbered questions 1 and 2 before the target question.',
                shortcut='Heading sequence predicts the number, not the question content.'),
    25015: dict(target='Go-Green', start=2308, alternative='Greenfield',
                mechanism='organization_identity', predictability='visible_identity',
                evidence='The contracting company was introduced in the page header and prior service sections.',
                shortcut='The company name has been repeated, including nearby.'),
    25043: dict(target='Eli Brookner', start=1812, alternative='Raoul Ouedraogo',
                mechanism='section_biography', predictability='visible_identity',
                evidence='The lecture header names the speaker; a later section starts his biography.',
                shortcut='The name is present in the header and nearby section transition.'),
}


def build(preview_json, manifest_json):
    rows = json.loads(Path(preview_json).read_text())
    manifest = json.loads(Path(manifest_json).read_text())
    assert [r['index'] for r in rows] == list(range(25000, 25048))
    hashes = {r['offset']: r['sha256'] for r in manifest['hashes']}
    eligible = [r for r in rows if r['full_length'] >= 2500]
    selected = random.Random(SEED).sample(eligible, 10)
    assert {r['index'] for r in selected} == set(SITES)
    sites = []
    for row in selected:
        spec = SITES[row['index']]
        start, target = spec['start'], spec['target']
        assert row['sha256'] == hashes[row['index']]
        assert row['preview'][start:start+len(target)] == target
        assert spec['alternative'] != target
        sites.append(dict(source_row=row['index'],
                          source_sha256=row['sha256'],
                          start_char=start, end_char=start+len(target), **{
                              k: v for k, v in spec.items() if k != 'start'}))
    return dict(source='HuggingFaceFW/fineweb sample-10BT train',
                selection='seed 25017 selects 10 of 16 >=2500-character '
                          'documents in independent offsets 25000..25047; '
                          'one manually selected target per document',
                caveat='Target selection within documents is purposive, not '
                       'a task prevalence sample; one reader and several '
                       'copy/topic shortcuts. Evidence is paraphrased.',
                sites=sites)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preview-json', type=Path, required=True)
    p.add_argument('--manifest-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = build(args.preview_json, args.manifest_json)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"pinned {len(result['sites'])} long target spans")
