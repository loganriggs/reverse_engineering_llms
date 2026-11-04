"""Semi-parametric joint rule DGP over distinct sampled source documents.

The source text is drawn from an empirical corpus. All rule gates and edits
are sampled here with recorded provenance. This gives local rule targets and
paired causal edits, but no symbolic oracle for untouched background tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from corpus_rule_reference_dgp import Document, sample_document


FAMILIES = ('list', 'bracket', 'literal_copy', 'natural_style')


@dataclass(frozen=True)
class Config:
    gate_rate: float = .5
    coupling_strength: float = 0.
    source_policy: str = 'uniform'
    excluded_pair: tuple[str, str] | None = None

    def __post_init__(self):
        assert 0 <= self.gate_rate <= 1
        assert 0 <= self.coupling_strength <= min(
            self.gate_rate, 1-self.gate_rate)
        assert self.source_policy in ('uniform', 'multitask')
        assert (self.excluded_pair is None or
                (len(self.excluded_pair) == 2 and
                 len(set(self.excluded_pair)) == 2 and
                 set(self.excluded_pair) <= set(FAMILIES)))


@dataclass(frozen=True)
class Draw:
    source_index: int
    source_sha256: str
    document: Document
    rich_state: int
    conditional_gate_rate: float
    enabled_gates: tuple[str, ...]
    gate_rejections: int


def eligibility(source_text: str):
    doc = sample_document(source_text, random.Random(0), (),
                          list_rate=0., bracket_rate=0., nonce_rate=0.,
                          literal_copy_rate=0., natural_style_rate=0.)
    return frozenset(doc.eligible)


def draw(source_row, rng: random.Random, config: Config):
    eligible = eligibility(source_row['preview'])
    # Draw every gate before candidate selection. When a whole-combination
    # holdout is requested, condition the law by rejection; record how often
    # rejection occurred rather than pretending marginal rates are unchanged.
    rejections = 0
    while True:
        rich = rng.choice((-1, 1)) if config.coupling_strength else 0
        rate = config.gate_rate + rich*config.coupling_strength
        enabled = frozenset(family for family in FAMILIES
                            if rng.random() < rate)
        if (config.excluded_pair is None or
                not set(config.excluded_pair) <= (enabled & eligible)):
            break
        rejections += 1
        if rejections >= 1000:
            raise RuntimeError('Too many excluded-pair gate rejections')
    doc = sample_document(
        source_row['preview'], rng, (),
        list_rate=float('list' in enabled),
        bracket_rate=float('bracket' in enabled),
        nonce_rate=0., list_shift_rate=.5,
        square_rate=.5,
        literal_copy_rate=float('literal_copy' in enabled),
        natural_style_rate=float('natural_style' in enabled))
    assert set(doc.applied) <= enabled
    return Draw(source_row['index'], source_row['sha256'], doc, rich, rate,
                tuple(sorted(enabled)), rejections)


def select_rows(source_rows, n: int, rng: random.Random, policy: str):
    assert policy in ('uniform', 'multitask')
    unique = {}
    for row in source_rows:
        unique.setdefault(row['sha256'], row)
    rows = list(unique.values())
    if policy == 'multitask':
        rows = [row for row in rows
                if len(eligibility(row['preview']) & set(FAMILIES)) >= 2]
    if n > len(rows):
        raise ValueError(f'Only {len(rows)} distinct eligible sources for {n} draws')
    return rng.sample(rows, n), len(rows)
