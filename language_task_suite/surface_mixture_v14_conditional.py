"""V14: completed item facts may continue into condition-driven actions.

The conditional action is local and should be an easier competency control
than V13's cross-sentence item-attribute binding. Both occur in the same
sampled stream, and a conditional clause remains a visible attribute fact
for later descriptive reuse. This remains a synthetic language pilot.
"""

from surface_mixture_v2 import PEOPLE
from surface_mixture_v13_attributes import (
    ATTRIBUTES, SurfaceMixtureV13Attributes, VOCAB_V13,
)


ACTION = dict(clean='uses', dusty='cleans', broken='repairs',
              valuable='protects', rare='inspects', heavy='moves')
assert set(ACTION) == set(ATTRIBUTES)
VOCAB_V14 = tuple(dict.fromkeys(VOCAB_V13 + ('so',) + tuple(ACTION.values())))
TOKEN2ID_V14 = {token:i for i,token in enumerate(VOCAB_V14)}
ID2TOKEN_V14 = {i:token for token,i in TOKEN2ID_V14.items()}


class SurfaceMixtureV14Conditional(SurfaceMixtureV13Attributes):
    def __init__(self, seed=0, conditional_continuation_rate=.25, **kwargs):
        super().__init__(seed=seed, **kwargs)
        assert 0 <= conditional_continuation_rate < 1
        self.conditional_continuation_rate=conditional_continuation_rate
        self.vocab,self.token2id,self.id2token=(
            VOCAB_V14,TOKEN2ID_V14,ID2TOKEN_V14)

    def _tokens(self,prefix):
        return [ID2TOKEN_V14[i] if isinstance(i,int) else i for i in prefix]

    def _fact_distribution(self,history,content):
        if len(content) <= 3 or content[2] == 'costs':
            return super()._fact_distribution(history,content)
        assert content[2] == 'is'
        pos=len(content)
        if pos==4:
            rate=self.conditional_continuation_rate
            return {'.':1-rate,',':rate}
        if pos==5:
            return {'so':1.}
        if pos==6:
            return {person:1/len(PEOPLE) for person in PEOPLE}
        if pos==7:
            return {ACTION[content[3]]:1.}
        if pos==8:
            return {'it':1.}
        assert pos==9
        return {'.':1.}

    def _base_label(self,words,content):
        if (content and content[0]=='The' and content[2:3]==['is']
                and len(content)==7 and content[4:6]==[',','so']):
            return 'conditional_action'
        return super()._base_label(words,content)
