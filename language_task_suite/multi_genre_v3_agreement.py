"""Multi-genre v3 with exact subject-verb agreement across an attractor.

Two sentence forms separate the main subject from the nearest noun:
  The pilots near the doctor are ready.
  The pilots that the doctor helps are ready.
The relative-clause verb agrees with the attractor; the main verb agrees
with the earlier subject. Both noun numbers are sampled independently.
"""

from multi_genre_language import _split_sentences
from multi_genre_v2_language import (GENRES_V2, VOCAB_V2,
    MultiGenreV2Language, render as render_v2)


GENRES_V3 = GENRES_V2 + ('Agreement',)
GENRE_WEIGHTS_V3 = (.28, .14, .14, .14, .15, .15)
NOUN_PAIRS = (('pilot', 'pilots'), ('artist', 'artists'),
              ('driver', 'drivers'), ('teacher', 'teachers'),
              ('writer', 'writers'), ('doctor', 'doctors'),
              ('gardener', 'gardeners'), ('visitor', 'visitors'))
NOUNS = tuple(noun for pair in NOUN_PAIRS for noun in pair)
NOUN_NUMBER = {noun: number for singular, plural in NOUN_PAIRS
               for number, noun in ((0, singular), (1, plural))}
NOUN_OTHER_NUMBER = {singular:plural for singular,plural in NOUN_PAIRS}
NOUN_OTHER_NUMBER.update({plural:singular for singular,plural in NOUN_PAIRS})
ADJECTIVES = ('ready', 'quiet', 'late', 'careful',
              'tired', 'calm', 'alert', 'busy')
TEMPLATES = {'near': .3, 'beside': .3, 'that': .4}
EXTRA = ('Agreement',) + NOUNS + ADJECTIVES + tuple(TEMPLATES) + (
    'The', 'the', 'is', 'are', 'helps', 'help')
VOCAB_V3 = tuple(dict.fromkeys(VOCAB_V2 + EXTRA))
TOKEN2ID_V3 = {word:i for i,word in enumerate(VOCAB_V3)}
ID2TOKEN_V3 = {i:word for word,i in TOKEN2ID_V3.items()}


def uniform(words):
    return {word:1/len(words) for word in words}


class MultiGenreV3Agreement(MultiGenreV2Language):
    def __init__(self, seed=0, genre_weights=GENRE_WEIGHTS_V3,
                 record_rate=.60):
        assert len(genre_weights) == len(GENRES_V3)
        assert min(genre_weights) >= 0 and sum(genre_weights) > 0
        parent_weights = (genre_weights[:5] if sum(genre_weights[:5]) else
                          (1., 0., 0., 0., 0.))
        super().__init__(seed=seed,genre_weights=parent_weights,
                         record_rate=record_rate)
        self.genre_weights = tuple(genre_weights)
        self.vocab = VOCAB_V3
        self.token2id = TOKEN2ID_V3
        self.id2token = ID2TOKEN_V3

    def _agreement(self, body):
        _, clause = _split_sentences(body)
        pos = len(clause)
        if pos == 0:
            return {'The':1.}, 'agreement_frame'
        if pos == 1:
            return uniform(NOUNS), 'agreement_subject'
        if pos == 2:
            return dict(TEMPLATES), 'agreement_frame'
        if pos == 3:
            return {'the':1.}, 'agreement_frame'
        if pos == 4:
            return uniform(NOUNS), 'agreement_attractor'
        relative = clause[2] == 'that'
        main_number = NOUN_NUMBER[clause[1]]
        attractor_number = NOUN_NUMBER[clause[4]] if pos > 4 else None
        if relative and pos == 5:
            return ({'help':1.} if attractor_number else {'helps':1.}), \
                   'agreement_embedded'
        main_position = 6 if relative else 5
        if pos == main_position:
            label = ('agreement_main_conflict' if
                     main_number != attractor_number else
                     'agreement_main_match')
            return ({'are':1.} if main_number else {'is':1.}), label
        if pos == main_position+1:
            return uniform(ADJECTIVES), 'agreement_predicate'
        if pos == main_position+2:
            return {'.':1.}, 'agreement_frame'
        raise ValueError(f'Bad Agreement prefix position {pos}')

    def next_distribution(self, prefix):
        words = [ID2TOKEN_V3[token] if isinstance(token,int) else token
                 for token in prefix]
        if not words:
            mass = sum(self.genre_weights)
            return ({genre:weight/mass for genre,weight in
                     zip(GENRES_V3,self.genre_weights) if weight}, 'genre')
        if words[0] != 'Agreement':
            return super().next_distribution(words)
        if len(words) == 1:
            return {':':1.}, 'header'
        assert words[1] == ':'
        return self._agreement(words[2:])

    def sample_sequence(self, length=72):
        words,labels,distributions = [],[],[]
        for _ in range(length):
            q,label = self.next_distribution(words)
            word = self.rng.choices(list(q),weights=list(q.values()))[0]
            words.append(word)
            labels.append(label)
            distributions.append(q)
        return [self.token2id[word] for word in words],labels,distributions


def render(ids):
    return render_v2([ID2TOKEN_V3[i] if isinstance(i,int) else i for i in ids])
