"""Two-site joint-rule language: competing verb and object mechanisms.

The v2 verb oracle is retained. Each object's next-token distribution also
mixes a local adjective bigram, previous same-subject object induction, a
frame cue, and a previous-sentence event skip. All choices are determined by
the visible prefix; no sampled latent rule is hidden from the oracle.
"""

from collections import defaultdict

from surface_mixture_v2 import (ADJECTIVES, CUES, EVENTS, OBJECTS,
                                SurfaceMixtureV2, _frame, _indices, _split,
                                TOKEN2ID)


OBJECT_INDEX = {'lead': 6, 'subject': 9, 'at': 8, 'when': 9,
                'in_place': 7, 'during': 7}
EVENT_INDEX = {'lead': 12, 'subject': 4, 'at': 11, 'when': 2,
               'in_place': 10, 'during': 2}
OBJECT_WEIGHTS = dict(adjective_bigram=.40, object_induction=.25,
                      frame_cue=.20, event_skip=.15)


class SurfaceMixtureV3(SurfaceMixtureV2):
    def __init__(self, seed=0, weights=None, object_weights=None):
        super().__init__(seed=seed, weights=weights)
        self.object_weights = dict(OBJECT_WEIGHTS if object_weights is None
                                   else object_weights)
        assert set(self.object_weights) == set(OBJECT_WEIGHTS)
        assert min(self.object_weights.values()) > 0

    def object_candidates(self, prefix):
        history, clause = _split(self._tokens(prefix))
        frame = _frame(clause)
        object_i = OBJECT_INDEX[frame]
        if len(clause) != object_i:
            raise ValueError('Object candidates exist only at an object slot')
        subject_i, _, _, _ = _indices(frame)
        subject = clause[subject_i]
        adjective = clause[object_i-1]
        cue = clause[2] if frame == 'subject' else clause[0]
        adjective_base = (ADJECTIVES.index(adjective)*3+1) % len(OBJECTS)
        out = {
            'adjective_bigram': {
                OBJECTS[(adjective_base+j) % len(OBJECTS)]: p
                for j, p in enumerate((.7, .2, .1))},
            'frame_cue': {
                OBJECTS[(CUES.index(cue)*11+3) % len(OBJECTS)]: 1.0},
        }
        matches = [old for old in history
                   if old[_indices(_frame(old))[0]] == subject]
        if matches:
            old = matches[-1]
            out['object_induction'] = {old[OBJECT_INDEX[_frame(old)]]: 1.0}
        if history:
            old = history[-1]
            event = old[EVENT_INDEX[_frame(old)]]
            out['event_skip'] = {
                OBJECTS[(EVENTS.index(event)*13+5) % len(OBJECTS)]: 1.0}
        return out

    def next_distribution(self, prefix):
        _, clause = _split(self._tokens(prefix))
        if clause and len(clause) == OBJECT_INDEX[_frame(clause)]:
            candidates = self.object_candidates(prefix)
            total = sum(self.object_weights[name] for name in candidates)
            out = defaultdict(float)
            for name, dist in candidates.items():
                for token, mass in dist.items():
                    out[token] += self.object_weights[name]*mass/total
            return dict(out)
        return super().next_distribution(prefix)

    def sample_sequence(self, length=72):
        tokens, labels, qs = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            _, clause = _split(tokens)
            label = 'frame'
            if clause:
                frame = _frame(clause)
                if len(clause) == _indices(frame)[1]:
                    label = 'mixed_verb'
                elif len(clause) == OBJECT_INDEX[frame]:
                    label = 'mixed_object'
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(token)
        return [TOKEN2ID[token] for token in tokens], labels, qs
