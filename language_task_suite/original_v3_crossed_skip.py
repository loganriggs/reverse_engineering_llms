"""Joint v3 grammar with a genuinely two-cue skip-rule table.

Each of four adjective triggers occurs under each of four place anchors, and
the correct output changes with either cue. The rest of the original sampled
joint grammar, including the shared-content bracket repair, is unchanged.
This is a mechanistic control, not a natural-language realism claim.
"""

from collections import defaultdict

from generator_v3 import SKIP_BIGRAM_MAX_SKIP
from original_v3_shared_bracket import LanguageV3SharedBracket


class LanguageV3CrossedSkip(LanguageV3SharedBracket):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.skip_bigrams:
            return
        original = tuple(self.skip_bigrams.items())
        anchors = tuple(pair[0] for pair, _ in original)
        triggers = tuple(pair[1] for pair, _ in original)
        outputs = tuple(target for _, target in original)
        if not (len(anchors) == len(set(anchors)) == len(triggers)
                == len(set(triggers)) == len(outputs) == len(set(outputs)) == 4):
            raise ValueError('Expected the default four distinct v3 skip rules')
        self.crossed_skip_anchors = anchors
        self.crossed_skip_triggers = triggers
        self.crossed_skip_outputs = outputs
        # A cyclic Latin square: knowing either cue alone leaves all four
        # outputs possible. Preserve the original pair at (anchor0, trigger0).
        self.skip_bigrams = {
            (anchor, trigger): outputs[(i+j) % 4]
            for i, anchor in enumerate(anchors)
            for j, trigger in enumerate(triggers)
        }
        # The original trigger-boost sampler iterates this list and stops at
        # the first accepted trigger. Rotate the order across anchors so that
        # its positional bias does not always favor the same adjective.
        self.skip_anchor_triggers = defaultdict(list)
        for i, anchor in enumerate(anchors):
            self.skip_anchor_triggers[anchor] = list(triggers[i:]+triggers[:i])


def nearest_skip_anchor(gen, prefix):
    """Return the actual nearest matching anchor for a skip-rule query."""
    trigger = prefix[-1]
    for dist in range(2, min(SKIP_BIGRAM_MAX_SKIP+2, len(prefix)+1)):
        anchor = prefix[-dist]
        if (anchor, trigger) in gen.skip_bigrams:
            return len(prefix)-dist, anchor
    return None
