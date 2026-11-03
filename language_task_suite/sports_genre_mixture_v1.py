"""Two observation regimes over one held-out-safe sampled game world.

The retrospective report and live ledger are generated from the same
underlying game. Their rule sites remain genre-specific: a retrospective
run is usually *not* inferable from visible scores, while the live ledger
prints a score history. This is a DGP control, not a realism pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from sports_event_world_dgp import world_from_entry
from sports_event_world_renderer import render_world
from sports_live_ledger_dgp import ledger_pool, sample_document as live_document


@dataclass(frozen=True)
class MatchedGenrePair:
    source: str
    split: str
    game_id: str
    retrospective: object
    live: object


def sample_pair(rng: random.Random, source='rotowire', split='train',
                retrospective_run_rate=None, live_descriptions=True):
    """Sample two genre views of one real game, with distinct answer rules."""
    pool = ledger_pool()[source, split]
    if not pool:
        raise RuntimeError('No joined game eligible for both genres')
    entry, _ = rng.choice(pool)
    world = world_from_entry(rng, source, split, entry)
    retrospective = render_world(world, rng,
                                 run_rate=retrospective_run_rate)
    live = live_document(rng, source, split, live_descriptions,
                         game_id=world.game_id)
    assert live.latent['game_id'] == world.game_id
    return MatchedGenrePair(source, split, world.game_id,
                            retrospective, live)


def sample_document(rng: random.Random, source='rotowire', split='train',
                    live_rate=.25, retrospective_run_rate=None,
                    live_descriptions=True):
    """Draw a single genre from the fixed mixture, returning genre and pair.

    The paired alternative is returned for controlled evaluation and must
    not be included in ordinary training unless that is the stated regime.
    """
    if not 0 <= live_rate <= 1:
        raise ValueError('live_rate must be between 0 and 1')
    pair = sample_pair(rng, source, split, retrospective_run_rate,
                       live_descriptions)
    genre = 'live' if rng.random() < live_rate else 'retrospective'
    return genre, pair.live if genre == 'live' else pair.retrospective, pair
