"""Paired form, site coverage, and rule audit for the V16 fact bank."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_document_world_v5_clause_realizations import _edited_world
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_surface_repetition import WORD
from compare_v12_fineweb_fresh import first_words
from document_world_v10_fact_acts import (
    answer_from_visible_prefix as answer_v10, realize as realize_v10)
from document_world_v14_variable_graph import draw_graph
from document_world_v16_proposition_bank import (
    answer_from_visible_prefix as answer_v16, extra_sentences,
    realize as realize_v16)


def site_window(doc, site, width):
    text = doc.text()
    words = list(WORD.finditer(text))
    if len(words) < width:
        return None
    target = len(WORD.findall(' '.join(doc.tokens[:site.index])))
    begin = max(0, min(target-width//2, len(words)-width))
    return text[words[begin].start():words[begin+width-1].end()]


def check(doc, parser):
    for site in doc.sites:
        assert doc.tokens[site.index] == site.answer
        assert parser(doc.tokens[:site.index], lexical_enabled=True) == site.answer


def audit(n, seed, words, width, n_ctx, tokenizer):
    rng = random.Random(seed)
    text = defaultdict(list)
    windows = defaultdict(list)
    counts = Counter()
    examples = []
    for _ in range(n):
        world = draw_graph(rng)
        base = realize_v10(world, lexical_enabled=True,
                           syntax_entropy=1., aside_rate=.3,
                           context_rate=.35)
        varied = realize_v16(world, lexical_enabled=True,
                             syntax_entropy=1., aside_rate=.3,
                             context_rate=.35)
        check(base, answer_v10)
        check(varied, answer_v16)
        assert [(s.family, s.answer) for s in base.sites] == [
            (s.family, s.answer) for s in varied.sites]
        counts.update('act/'+kind for kind, _ in extra_sentences(world).values())
        counts.update('site/'+site.family for site in varied.sites)
        for name, doc in (('base', base), ('v16', varied)):
            crop = first_words(doc.text(), words)
            if len(WORD.findall(crop)) >= words:
                text[name].append(crop)
            source = next((s for s in doc.sites
                           if s.family == 'source_attribution'), None)
            if source is not None:
                window = site_window(doc, source, width)
                if window:
                    windows[name].append(window)
            for site in doc.sites:
                prefix = replace(doc, tokens=doc.tokens[:site.index]).text()
                if len(tokenizer.encode(prefix, add_special_tokens=False).ids) < n_ctx:
                    counts[name+'/site_under_ctx'] += 1
        changed_world = _edited_world(world, world['recap_plan'][0])
        changed = realize_v16(changed_world, lexical_enabled=True,
                              syntax_entropy=1., aside_rate=.3,
                              context_rate=.35)
        check(changed, answer_v16)
        assert changed.sites[0].answer != varied.sites[0].answer
        counts['supported_edit_checks'] += 1
        if len(examples) < 2:
            examples.append(varied.text()[:700])
    return {'counts': dict(counts),
            'first100_form': {key: structure(rows) for key, rows in text.items()},
            'first100_counts': {key: len(rows) for key, rows in text.items()},
            'source41_form': {key: structure(rows)
                              for key, rows in windows.items()},
            'source41_counts': {key: len(rows)
                                for key, rows in windows.items()},
            'examples': examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--seed', type=int, default=56701)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--window-words', type=int, default=41)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    panels = {str(a.seed+i): audit(a.n, a.seed+i, a.words,
                                   a.window_words, a.n_ctx, tokenizer)
              for i in range(2)}
    result = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'panels': panels,
              'scope': 'Paired worlds; same scored sites and answers. '
                       'Visible-prefix checks and one supported edit per '
                       'world. First100 and 41-word source-site form. '
                       'Context coverage uses BPE token count of rendered '
                       'prefix, approximate by at most a boundary token. '
                       'No full next-token oracle or natural-text claim.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for seed, row in panels.items():
        print(seed, row['counts'], flush=True)
        for panel in ('first100_form', 'source41_form'):
            for name, form in row[panel].items():
                print(panel, name, 'types', form['word_types'],
                      'top10', round(form['top10_word_share'], 3),
                      'repeat', round(form[
                          'repeated_across_docs_trigram_occurrence_share'], 3),
                      flush=True)


if __name__ == '__main__':
    main()
