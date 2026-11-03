"""Contrastive source claims in the same joint document grammar.

Reference and task-rich regimes have separate, exact visible-prefix
distributions. V2 makes speaker collisions possible and balances query keys
before any model is trained; it does not claim natural-language realism.
"""

import random

from joint_document_v1 import (ATTRIBUTES, NAMES, _determiners, _normalize,
                               _split, _zipf,
                               next_distribution as base_distribution,
                               site_label, visible_state)


def _hard_items(claims):
    items = dict.fromkeys(item for item, _ in claims)
    return tuple(item for item in items if len({speaker for (key, _), speaker
                                               in claims.items() if key == item}) >= 2)


def _blend(base, candidates, mass):
    available = tuple(word for word in candidates if word in base)
    if not available or not mass:
        return base
    out = {word: (1-mass)*p for word, p in base.items()}
    for word in available:
        out[word] += mass/len(available)
    return _normalize(out)


def next_distribution(prefix, regime='reference'):
    assert regime in ('reference', 'task_rich')
    base = base_distribution(prefix)
    history, raw = _split(prefix)
    state = visible_state(history)
    rich = regime == 'task_rich'
    known = tuple(state['profiles'])
    claims = state['claims']
    if not raw and state['genre'] is not None:
        out = dict(base)
        if len(known) == 1:
            out = _blend(out, (name for name in NAMES if name not in known),
                         .26 if rich else .12)
        if len(known) >= 2:
            distinct_speakers = len(set(claims.values()))
            hard = _hard_items(claims)
            if (hard if rich else distinct_speakers >= 2):
                out = _blend(out, ('Who',), .26 if rich else .08)
            else:
                out.pop('Who', None)
                out = _normalize(out)
                out = _blend(out, ('According', 'As'),
                             .30 if rich else .10)
        else:
            out.pop('Who', None)
            out = _normalize(out)
        return out
    if not raw or raw[0] not in ('According', 'As', 'Who'):
        return base
    first, pos = raw[0], len(raw)
    if first in ('According', 'As'):
        speaker_pos = 2 if first == 'According' else 1
        if pos == speaker_pos:
            usage = {name: 0 for name in known}
            for speaker in claims.values():
                usage[speaker] += 1
            least = min(usage.values())
            return _blend(base, (name for name in known
                                 if usage[name] == least), .72 if rich else .35)
        if pos == 5:
            speaker = raw[speaker_pos]
            prior_other = tuple(dict.fromkeys(item for (item, _), source
                                              in claims.items() if source != speaker))
            return _blend(base, prior_other, .68 if rich else .18)
        if pos == 7:
            item = raw[5]
            unseen = [attr for attr in ATTRIBUTES if (item, attr) not in claims]
            return _blend(base, unseen, .80 if rich else .35)
    if first == 'Who' and rich:
        hard = _hard_items(claims)
        if pos == 2 and hard:
            return _determiners(_zipf(hard, .25))
        if pos == 3 and hard:
            return _normalize({item: p for item, p in base.items()
                               if item in hard})
        if pos == 5 and raw[3] in hard:
            return _zipf(tuple(attr for (item, attr) in claims
                               if item == raw[3]), .25)
    return base


def sample_sequence(seed, length=256, regime='reference'):
    rng = random.Random(seed)
    words, labels, distributions = [], [], []
    for _ in range(length):
        q = next_distribution(words, regime)
        label = site_label(words)
        word = rng.choices(tuple(q), weights=tuple(q.values()))[0]
        assert q[word] > 0 and abs(sum(q.values())-1) < 1e-10
        words.append(word)
        labels.append(label)
        distributions.append(q)
    return words, labels, distributions
