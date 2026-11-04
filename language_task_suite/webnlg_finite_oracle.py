"""Exact finite-support token distribution over original enriched WebNLG text.

This is a reference baseline only: the support is a fixed corpus of human
realizations. It cannot certify compositional generalization or realistic
document structure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from tokenizers import Tokenizer

from audit_enriched_webnlg_reuse import DEFAULT_ROOT
from audit_fineweb_surfaces import DEFAULT_TOKENIZER


@dataclass(frozen=True)
class Entry:
    category: str
    triples: tuple[str, ...]
    texts: tuple[str, ...]


class Node:
    __slots__ = ('mass', 'support', 'children')

    def __init__(self):
        self.mass = 0.
        self.support = 0
        self.children: dict[int, Node] = {}


def load_entries(root=DEFAULT_ROOT):
    by_category = defaultdict(list)
    for path in sorted(Path(root).glob('*triples/*.xml')):
        for item in ET.parse(path).findall('.//entry'):
            texts = tuple((lex.findtext('text') or '').strip()
                          for lex in item.findall('./lex')
                          if lex.get('comment') == 'good' and
                          (lex.findtext('text') or '').strip())
            if not texts:
                continue
            triples = tuple((node.text or '').strip() for node in
                            item.findall('./modifiedtripleset/mtriple'))
            by_category[item.get('category', '')].append(
                Entry(item.get('category', ''), triples, texts))
    return dict(by_category)


class FiniteOracle:
    def __init__(self, entries, tokenizer, *, category_exponent=0.):
        if not 0 <= category_exponent <= 1:
            raise ValueError('category_exponent must be in [0,1]')
        self.entries = entries
        self.tokenizer = tokenizer
        self.eos = tokenizer.token_to_id('<|endoftext|>')
        if self.eos is None:
            raise ValueError('Tokenizer has no EOS token')
        self.categories = sorted(entries)
        powers = [len(entries[c])**category_exponent
                  for c in self.categories]
        total = sum(powers)
        self.category_probs = {c: power/total
                               for c, power in zip(self.categories, powers)}
        self.root = Node()
        self.items = []
        self.node_count = 1
        for category in self.categories:
            for entry in entries[category]:
                weight = (self.category_probs[category]/
                          len(entries[category])/len(entry.texts))
                for text in entry.texts:
                    ids = tuple(tokenizer.encode(text, add_special_tokens=False).ids)
                    sequence = ids+(self.eos,)
                    self.items.append((category, sequence, weight, text))
                    node = self.root
                    node.mass += weight
                    node.support += 1
                    for token in sequence:
                        if token not in node.children:
                            node.children[token] = Node()
                            self.node_count += 1
                        node = node.children[token]
                        node.mass += weight
                        node.support += 1

    def sample(self, rng: random.Random):
        category = rng.choices(self.categories,
                               [self.category_probs[c]
                                for c in self.categories])[0]
        entry = rng.choice(self.entries[category])
        text = rng.choice(entry.texts)
        return entry, text

    def node(self, prefix):
        node = self.root
        for token in prefix:
            node = node.children.get(token)
            if node is None:
                return None
        return node

    def next_distribution(self, prefix):
        node = self.node(prefix)
        if node is None or not node.children:
            return {}
        return {token: child.mass/node.mass
                for token, child in node.children.items()}

    def prefix_stats(self, prefix):
        node = self.node(prefix)
        if node is None:
            return None
        p = self.next_distribution(prefix)
        return {'support': node.support, 'branches': len(p),
                'entropy_bits': -sum(v*math.log2(v) for v in p.values())}


def build(*, root=DEFAULT_ROOT, tokenizer_path=DEFAULT_TOKENIZER,
          category_exponent=0.):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return FiniteOracle(load_entries(root), tokenizer,
                        category_exponent=category_exponent)
