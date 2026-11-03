"""Natural-report form control with an event-verified run insertion.

The human carrier is empirical text. Only the inserted score snapshots and
run-answer sites have explicit generated semantics. This is a control, not
the desired fully sampled language DGP.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random
import re

from audit_rotowire_document_source import game_key, score_pair_present
from audit_sports_pbp_join import metadata, report_key
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import (Site, _detokenize, game_partition,
                                 resources)
from sports_event_world_dgp import ARCHIVE_DIR, world_from_entry, world_pool
from sports_event_world_renderer import run_sentences


@dataclass(frozen=True)
class CarriedDocument:
    text: str
    original: str
    sites: tuple[Site, ...]
    game_id: str
    source_game_key: tuple


@lru_cache(maxsize=8)
def eligible_event_carriers(source, group, carrier_words=100):
    meta = metadata(ARCHIVE_DIR/'shotdetail_2016.tar.xz')
    by_key = {}
    for entry in world_pool()[source, group]:
        game_id, home, away, final = entry[:4]
        item = meta[game_id]
        key = item['date'], frozenset((home, away)), final
        by_key[key] = entry
    out = []
    for row in resources()[0][source][0]:
        key = game_key(row)
        if game_partition(key) != group or \
                len(row['summary']) < carrier_words or \
                not score_pair_present(row):
            continue
        entry = by_key.get(report_key(row))
        if entry:
            out.append((key, row, entry))
    return tuple(out)


def realize(rng: random.Random, source='rotowire', group='train',
            carrier_words=100):
    candidates = eligible_event_carriers(source, group, carrier_words)
    if not candidates:
        raise RuntimeError('No event-linked natural carriers')
    key, row, entry = rng.choice(candidates)
    world = world_from_entry(rng, source, group, entry)
    block = run_sentences(world, rng)
    if block is None:
        raise RuntimeError('No regulation run for carrier')
    original_sentences = [_detokenize(tokens) for tokens in
                          _sentences(row['summary'])]
    if len(original_sentences) < 3:
        raise RuntimeError('Carrier too short')
    original = ' '.join(original_sentences).replace('*NEWPARAGRAPH*', '\n\n')
    # Preserve the source report and insert after an early sentence. The
    # matched final score may appear anywhere, not necessarily in the lead.
    slot = rng.randint(1, min(4, len(original_sentences)-1))
    sentences = original_sentences[:slot]+list(block)+ \
        original_sentences[slot:]
    text = ' '.join(sentences).replace('*NEWPARAGRAPH*', '\n\n')
    third = block[2]
    offset = text.index(third)
    pair = re.search(r'\b(\d+)-(\d+) run\b', third)
    assert pair
    sites = []
    for kind, number in (('run_for', 1), ('run_against', 2)):
        start, end = offset+pair.start(number), offset+pair.end(number)
        sites.append(Site(kind, start, end, text[start:end],
                          tuple(map(str, range(41)))))
    return CarriedDocument(text, original, tuple(sites), entry[0], key)
