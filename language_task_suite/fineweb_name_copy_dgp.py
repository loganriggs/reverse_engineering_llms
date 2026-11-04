"""Corpus-carried, sampled surname copying with known edit targets.

The source passages are external FineWeb data and are never checked in.
This is an exploratory *conditional* DGP: it samples only passages with a
repeated eligible capitalized token. It does not provide a full-token oracle.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import random
import re


NAME = re.compile(r'(?<![\w])([A-Z][a-z]{2,})(?![\w])')
FULL_PERSON = re.compile(r'[A-Z][a-z]+ ([A-Z][a-z]+)')
SURNAMES = tuple('''Smith Johnson Williams Jones Miller Davis Wilson Moore
Taylor Anderson Jackson Harris Martin Thompson Garcia Martinez Robinson Clark
Rodriguez Lewis Lee Walker Hall Allen Young King Wright Scott Torres Nguyen
Hill Green Adams Baker Nelson Carter Mitchell Perez Roberts Turner Phillips
Campbell Parker Evans Edwards Collins Stewart Sanchez Morris Rogers Reed Cook
Morgan Bell Murphy Bailey Rivera Cooper Richardson Howard Ward Peterson Gray
Ramirez Watson Brooks Kelly Sanders Bennett Barnes Ross Henderson Coleman
Jenkins Perry Powell Patterson Hughes Foster Bryant Alexander Russell Griffin
Diaz Hayes Myers Ford Hamilton Graham Sullivan Wallace Woods Cole West Jordan
Owens Reynolds Fisher Ellis Harrison Gibson McDonald Cruz Marshall Ortiz Gomez
Murray Freeman Wells Webb Simpson Stevens Tucker Porter Hunter Hicks Crawford
Henry Boyd Mason Morales Kennedy Warren Dixon Ramos Reyes Burns Gordon Shaw
Holmes Rice Robertson Hunt Black Daniels Palmer Mills Nichols Grant Knight
Ferguson Rose Stone Hawkins Dunn Perkins Hudson Spencer Gardner Stephens
Payne Pierce Berry Matthews Arnold Wagner Willis Ray Watkins Olson Carroll
Duncan Snyder Hart Cunningham Bradley Lane Andrews Ruiz Harper Fox Riley
Armstrong Carpenter Weaver Greene Lawrence Elliott Chavez Sims Austin Peters'''.split())
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EXCLUDE = frozenset('''A An And As At After Before But By Can Could Did Do
Does Each For From Had Has Have He Her His How I If In Into Is It Its Just
May More Most My No Not Of On Or Our Please She So Some That The Their Them
There These They This Those To Today Tomorrow Was We Were What When Where
Which While Who Why Will With Would You Your Monday Tuesday Wednesday
Thursday Friday Saturday Sunday January February March April May June July
August September October November December County University Committee
School State City Department Company Government United National Great New
Now First Last One Two Three All Any Every Here Then Also Since According
However Read See Learn Click Home Page Contact Terms Privacy Status Daddy
Christ God Lord'''.split())


@dataclass(frozen=True)
class Sample:
    text: str
    source_index: int
    source_sha256: str
    original: str
    replacement: str
    sites: tuple[tuple[int, int], ...]  # replacement spans after first mention
    candidate_count: int
    replacement_count: int


def _in_url(text: str, start: int) -> bool:
    left = text[max(0, start-80):start]
    tail = re.split(r'\s', left)[-1]
    return ('://' in tail or 'www.' in tail or '@' in tail or
            '/' in tail or '\\' in tail)


def name_candidates(text: str, min_gap_chars: int = 60):
    """Conservative regex opportunities, not verified person/entity tags."""
    locations = defaultdict(list)
    for match in NAME.finditer(text):
        word = match.group(1)
        if word in EXCLUDE or _in_url(text, match.start()):
            continue
        locations[word].append(match.start())
    result = {}
    for word, positions in locations.items():
        if len(positions) < 2 or positions[-1]-positions[0] < min_gap_chars:
            continue
        # Some capitalization is needed beyond a heading or sentence start.
        if not any(text[:pos].rstrip()[-1:] not in ('', '.', '?', '!', '\n')
                   for pos in positions):
            continue
        # A lowercase use of the same spelling would be left inconsistent.
        if re.search(r'(?<!\w)'+re.escape(word.lower())+r'(?!\w)', text):
            continue
        result[word] = tuple(positions)
    return result


class NameCopyDGP:
    def __init__(self, rows, min_gap_chars=60, max_source_doc_frequency=None,
                 mode='full_person'):
        if mode not in ('full_person', 'broad_regex'):
            raise ValueError(mode)
        self.rows = tuple(rows)
        self.min_gap_chars = min_gap_chars
        self.mode = mode
        if mode == 'full_person':
            import spacy
            from faker.providers.person.en import Provider
            given_names = (set(Provider.first_names_male) |
                           set(Provider.first_names_female))
            nlp = spacy.load('en_core_web_sm', disable=(
                'parser', 'tagger', 'attribute_ruler', 'lemmatizer'))
            raw, pool = [], set(SURNAMES)
            for row, doc in zip(self.rows, nlp.pipe(
                    [item['preview'] for item in self.rows]), strict=True):
                text = row['preview']
                full_names = Counter()
                for entity in doc.ents:
                    if entity.label_ != 'PERSON':
                        continue
                    match = FULL_PERSON.fullmatch(entity.text)
                    if match is None or match.group(1) in EXCLUDE:
                        continue
                    first_name = entity.text.split()[0]
                    if first_name in ('Saint', 'St', 'Mac') or \
                            first_name not in given_names:
                        continue
                    surname = match.group(1)
                    full_names[entity.text] += 1
                group = {}
                for full_name, count in full_names.items():
                    surname = FULL_PERSON.fullmatch(full_name).group(1)
                    if _in_url(text, text.index(full_name)) or re.search(
                            r'(?<!\w)'+re.escape(surname.lower())+r'(?!\w)',
                            text):
                        continue
                    positions = tuple(m.start() for m in re.finditer(
                        r'(?<!\w)'+re.escape(surname)+r'(?!\w)', text))
                    associated = {m.group(1) for m in re.finditer(
                        r'\b([A-Z][a-z]+)\s+'+re.escape(surname)+r'\b',
                        text) if m.group(1) in given_names}
                    if len(associated) > 1:
                        continue
                    first_full = text.index(full_name)+len(
                        full_name.split()[0])+1
                    if any(_in_url(text, pos) for pos in positions):
                        continue
                    if len(positions) >= 2 and \
                            positions[-1]-positions[0] >= min_gap_chars and \
                            positions[-1] > first_full:
                        group[surname] = positions
                raw.append(group)
        else:
            raw = [name_candidates(row['preview'], min_gap_chars)
                   for row in self.rows]
            pool = set().union(*(set(group) for group in raw))
        frequency = Counter(word for group in raw for word in group)
        self.candidates = tuple({word: sites for word, sites in group.items()
                                 if max_source_doc_frequency is None or
                                 frequency[word] <= max_source_doc_frequency}
                                for group in raw)
        self.pool = tuple(sorted(pool))
        self.eligible = tuple(i for i, group in enumerate(self.candidates)
                              if group)
        if len(self.pool) < 2 or not self.eligible:
            raise ValueError('No eligible repeated-name substitutions')

    def replacements(self, index, original):
        text = self.rows[index]['preview']
        # Treat possessives such as "Martin's" as containing Martin. A
        # replacement already present anywhere in the passage would make
        # the reverse edit alter unrelated text.
        present = {part.lower() for word in WORD.findall(text)
                   for part in re.findall(r'[A-Za-z]+', word)}
        return tuple(word for word in self.pool if word != original and
                     word.lower() not in present)

    def make(self, index, original, replacement):
        if original not in self.candidates[index] or \
                replacement not in self.replacements(index, original):
            raise ValueError('Unsupported source or replacement')
        text = self.rows[index]['preview']
        matches = list(re.finditer(r'(?<!\w)'+re.escape(original)+r'(?!\w)',
                                  text))
        assert len(matches) == len(self.candidates[index][original])
        parts, sites, cursor, length = [], [], 0, 0
        for j, match in enumerate(matches):
            before = text[cursor:match.start()]
            parts.extend((before, replacement))
            length += len(before)
            if j:
                sites.append((length, length+len(replacement)))
            length += len(replacement)
            cursor = match.end()
        parts.append(text[cursor:])
        changed = ''.join(parts)
        assert all(changed[a:b] == replacement for a,b in sites)
        return Sample(changed, self.rows[index]['index'],
                      hashlib.sha256(text.encode()).hexdigest(), original,
                      replacement, tuple(sites),
                      len(self.candidates[index]),
                      len(self.replacements(index, original)))

    def sample(self, rng: random.Random):
        index = rng.choice(self.eligible)
        original = rng.choice(tuple(self.candidates[index]))
        replacement = rng.choice(self.replacements(index, original))
        return self.make(index, original, replacement)

    def counterfactual(self, sample: Sample, rng: random.Random):
        index = next(i for i in self.eligible if
                     self.rows[i]['index'] == sample.source_index)
        options = [word for word in self.replacements(index, sample.original)
                   if word != sample.replacement]
        return self.make(index, sample.original, rng.choice(options))
