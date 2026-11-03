"""V15 pilot: varied ordinary because-tails within the V14 sampled stream.

This changes surface realization, not the underlying task rules. It keeps an
exact next-token oracle by branching only at a visible token boundary.
"""

from surface_mixture_v2 import OBJECTS, _frame, _indices, _split, uniform
from surface_mixture_v4 import _content
from surface_mixture_v12_price_faq import SPECIAL
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional, VOCAB_V14


VOCAB_V15 = tuple(dict.fromkeys(VOCAB_V14 + ('needed','waiting','nearby')))
TOKEN2ID_V15 = {token:i for i,token in enumerate(VOCAB_V15)}
ID2TOKEN_V15 = {i:token for token,i in TOKEN2ID_V15.items()}
TAIL_BRANCH = {'had':.40,'wanted':.25,'needed':.20,'was':.15}


class SurfaceMixtureV15TailVariants(SurfaceMixtureV14Conditional):
    def __init__(self, seed=0, **kwargs):
        super().__init__(seed=seed,**kwargs)
        self.vocab,self.token2id,self.id2token=(
            VOCAB_V15,TOKEN2ID_V15,ID2TOKEN_V15)

    def _tokens(self,prefix):
        return [ID2TOKEN_V15[i] if isinstance(i,int) else i for i in prefix]

    def next_distribution(self,prefix):
        words=self._tokens(prefix)
        _,raw=_split(words)
        content=_content(raw)
        if content and content[0] not in SPECIAL:
            frame=_frame(content)
            suffix=content[_indices(frame)[3]:]
            if suffix[:1]==['because'] and len(suffix)>=2:
                if len(suffix)==2:
                    return dict(TAIL_BRANCH)
                branch=suffix[2]
                if branch!='had':
                    variants={
                        'wanted': ({'a':1.},{'written':1.},{'record':1.},{'.':1.}),
                        'needed': ({'the':1.},uniform(OBJECTS),{'.':1.}),
                        'was': ({'waiting':1.},{'nearby':1.},{'.':1.}),
                    }
                    return variants[branch][len(suffix)-3]
        return super().next_distribution(prefix)
