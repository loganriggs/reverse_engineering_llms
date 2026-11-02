"""Compare repeat/closure opportunity proxies with a frozen FineWeb panel.

These word-pattern statistics are not labels for latent computations. The
stopword filter is heuristic: repeated function words and fixed phrase tails
can dominate the unconditioned successor-copy rate. Outputs contain
aggregates and source hashes, never decoded FineWeb passages.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from surface_alias_mixture import AliasMixtureLanguage, render as render_alias
from surface_mixture_v2 import (SurfaceMixtureV2, _frame, _indices,
                                _metadata, _split, render as render_v2)
from surface_mixture_v2_varied import (SurfaceMixtureV2Varied,
                                       render as render_v2_varied)
from surface_mixture_v7 import render as render_v8
from surface_mixture_v8 import SurfaceMixtureV8


HERE = Path(__file__).resolve().parent
DEFAULT_PANEL = (HERE.parents[1]/'reverse_engineer/tensor_language/'
                 'basis_aligned/bilinear_quotient/.rowcache/'
                 'fineweb_n192_skip7000.pt')
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STOP = frozenset('''about after again against all also and any are because
been before being between both but can could did does doing down each for
from had has have here how into its just more most much not now off once
only other our out over own same she should some such than that the their
them then there these they this those through too under until very was were
what when where which while who will with would you your'''.split())


def first_words(text, n=100):
    matches = list(WORD.finditer(text))
    return text[:matches[n-1].end()] if len(matches) >= n else text


def content(word):
    return len(word) >= 4 and word not in STOP


def repeat_stats(texts):
    total_content = repeat = same_next = random_next = 0
    lexical_successor_opportunities = lexical_successor_copies = 0
    repeat_windows = 0
    distance_counts = Counter()
    for text in texts:
        words = [match.group().lower() for match in WORD.finditer(text)]
        last = {}
        found = False
        for i, word in enumerate(words):
            if not content(word):
                continue
            total_content += 1
            j = last.get(word)
            last[word] = i
            if j is None or i-j > 64 or i+1 >= len(words) or j+1 >= i:
                continue
            repeat += 1
            found = True
            distance = i-j
            distance_counts[('2-4' if distance <= 4 else
                             '5-16' if distance <= 16 else '17-64')] += 1
            same_next += words[i+1] == words[j+1]
            if content(words[i+1]):
                lexical_successor_opportunities += 1
                lexical_successor_copies += words[i+1] == words[j+1]
            # Deterministic within-window random-prior control. Its word differs
            # from the repeated word and it has a successor before position i.
            choices = [k for k in range(max(0, i-64), i-1)
                       if content(words[k]) and words[k] != word]
            if choices:
                k = choices[(i*131+j*17) % len(choices)]
                random_next += words[i+1] == words[k+1]
        repeat_windows += found
    return dict(content_tokens=total_content,
                repeat_opportunities=repeat,
                repeat_per_100_content=100*repeat/max(1,total_content),
                windows_with_repeat=repeat_windows,
                same_next_after_repeat_rate=same_next/max(1,repeat),
                lexical_successor_opportunities=lexical_successor_opportunities,
                lexical_successor_copy_rate=(lexical_successor_copies /
                                             max(1, lexical_successor_opportunities)),
                same_next_after_random_prior_rate=random_next/max(1,repeat),
                repeat_distance_counts=dict(distance_counts))


def span_stats(texts):
    lengths = {'parenthesis': [], 'square_bracket': []}
    windows = Counter()
    nested = Counter()
    unclosed = Counter()
    for text in texts:
        stack = []
        seen = set()
        for i, char in enumerate(text):
            if char in '([':
                kind = 'parenthesis' if char == '(' else 'square_bracket'
                stack.append((char, i))
                if len(stack) > 1:
                    nested[kind] += 1
            elif char in ')]':
                opener = '(' if char == ')' else '['
                kind = 'parenthesis' if char == ')' else 'square_bracket'
                if not stack or stack[-1][0] != opener:
                    unclosed[kind] += 1
                    continue
                _, start = stack.pop()
                lengths[kind].append(len(WORD.findall(text[start+1:i])))
                seen.add(kind)
        for opener, _ in stack:
            unclosed['parenthesis' if opener == '(' else 'square_bracket'] += 1
        windows.update(seen)
    return {kind: dict(windows_with_complete_span=windows[kind],
                       complete_spans=len(values),
                       median_length_words=float(np.median(values)) if values else None,
                       p90_length_words=float(np.percentile(values, 90)) if values else None,
                       nested_opens=nested[kind], unclosed_edges=unclosed[kind])
            for kind, values in lengths.items()}


def summaries(texts):
    cropped = [first_words(text) for text in texts]
    counts = [len(WORD.findall(text)) for text in cropped]
    return dict(n_windows=len(cropped), mean_words=float(np.mean(counts)),
                distinct_words=len(set(word.lower() for text in cropped
                                       for word in WORD.findall(text))),
                repeats=repeat_stats(cropped), spans=span_stats(cropped),
                form_windows=dict(newline=sum('\n' in t for t in cropped),
                                  question=sum('?' in t for t in cropped),
                                  colon=sum(':' in t for t in cropped),
                                  digit=sum(bool(re.search(r'\d', t))
                                            for t in cropped)))


def sample_v2(gen, n, length, render_fn=render_v2):
    texts = []
    verbs = induction_active = conflicting = 0
    induction_distances = []
    nonadjacent_sources = 0
    for _ in range(n):
        ids, labels, _ = gen.sample_sequence(length)
        texts.append(render_fn(ids))
        for position, label in enumerate(labels):
            if label != 'mixed_verb':
                continue
            verbs += 1
            candidates = gen.candidates(ids[:position])
            induction_active += 'induction' in candidates
            if 'induction' in candidates:
                history, clause = _split([gen.id2token[i]
                                          for i in ids[:position]])
                subject = clause[_indices(_frame(clause))[0]]
                sources = [i for i, old in enumerate(history)
                           if _metadata(old)['subject'] == subject]
                source_i = sources[-1]
                source_position = (sum(map(len, history[:source_i])) +
                                   _indices(_frame(history[source_i]))[1])
                induction_distances.append(position-source_position)
                nonadjacent_sources += source_i < len(history)-1
            modal = [max(dist, key=dist.get) for dist in candidates.values()]
            conflicting += len(set(modal)) > 1
    return texts, dict(verb_sites=verbs,
                       induction_active=induction_active,
                       induction_fraction=induction_active/max(1, verbs),
                       induction_distance_median=(float(np.median(induction_distances))
                                                  if induction_distances else None),
                       induction_distance_p90=(float(np.percentile(induction_distances, 90))
                                               if induction_distances else None),
                       nonadjacent_induction_fraction=(nonadjacent_sources /
                                                       max(1, induction_active)),
                       modal_conflict_sites=conflicting,
                       modal_conflict_fraction=conflicting/max(1, verbs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=Path, default=DEFAULT_PANEL)
    p.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument('--v3-source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--first-tokens', type=int, default=160)
    p.add_argument('--seed', type=int, default=20261115)
    p.add_argument('--output', type=Path, default=HERE/'dependency_proxy_audit_dev.json')
    a = p.parse_args()
    rows = torch.load(a.panel, map_location='cpu', weights_only=True)
    assert rows.ndim == 2 and rows.shape[1] >= a.first_tokens
    digest = hashlib.sha256(rows.numpy().tobytes()).hexdigest()
    known = json.loads((HERE/'fineweb_dependency_panel_skip7000.json').read_text())
    if a.panel == DEFAULT_PANEL:
        assert digest == known['source_sha256']
    elif a.panel.name == 'fineweb_n192_skip11000.pt':
        holdout = json.loads((HERE/'alias_surface_holdout.json').read_text())
        assert digest == holdout['holdout_sha256']
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    natural = [tokenizer.decode(row[:a.first_tokens].tolist(),
                                skip_special_tokens=False) for row in rows]
    sys.path.insert(0, a.v3_source)
    from generator_v3 import LanguageV3
    n = len(natural)
    v3 = LanguageV3(tokens_per_category=16, seed=a.seed)
    v3_texts = []
    v3_labels = Counter()
    for _ in range(n):
        ids, labels = v3.sample_sequence(a.first_tokens)
        v3_labels.update(labels)
        v3_texts.append(' '.join(v3.id2token[token] for token in ids))
    v3_sparse = LanguageV3(tokens_per_category=16, paren_prob=.006,
                           quote_prob=.0007, seed=a.seed)
    v3_sparse_texts = []
    v3_sparse_labels = Counter()
    for _ in range(n):
        ids, labels = v3_sparse.sample_sequence(a.first_tokens)
        v3_sparse_labels.update(labels)
        v3_sparse_texts.append(' '.join(v3_sparse.id2token[token]
                                       for token in ids))
    v2 = SurfaceMixtureV2(seed=a.seed)
    v2_texts, v2_density = sample_v2(v2, n, a.first_tokens)
    v2_repeat = {}
    v2_repeat_density = {}
    for mass in (.2, .1, 0.):
        generator = SurfaceMixtureV2(seed=a.seed,
                                     subject_repeat_mass=mass)
        v2_repeat[mass], v2_repeat_density[mass] = sample_v2(
            generator, n, a.first_tokens)
    return_texts, return_density = sample_v2(
        SurfaceMixtureV2(seed=a.seed, subject_repeat_mass=.1,
                         subject_return_mass=.3), n, a.first_tokens)
    varied_texts, varied_density = sample_v2(
        SurfaceMixtureV2Varied(seed=a.seed), n, a.first_tokens,
        render_fn=render_v2_varied)
    varied_low_texts, varied_low_density = sample_v2(
        SurfaceMixtureV2Varied(seed=a.seed, parenthetical_mass=.1),
        n, a.first_tokens, render_fn=render_v2_varied)
    v8 = SurfaceMixtureV8(seed=a.seed, section_rate=.06, detail_rate=.8,
                          scope_question_rate=.08, quote_rate=.08,
                          source_rate=.15)
    v8_texts = [render_v8(v8.sample_sequence(a.first_tokens)[0])
                for _ in range(n)]
    alias = AliasMixtureLanguage(seed=a.seed, parenthetical_rate=.25,
                                 definition_start_rate=.10,
                                 alias_reuse_rate=.10)
    alias_texts = [render_alias(alias.sample_sequence(a.first_tokens)[0])
                   for _ in range(n)]
    natural_name = ('fineweb_dev' if a.panel == DEFAULT_PANEL
                    else 'fineweb_holdout')
    candidates = [(natural_name, natural), ('v3_priority', v3_texts),
                  ('v3_sparse_brackets', v3_sparse_texts),
                  ('v2_joint', v2_texts)]
    candidates += [(f'v2_subject_repeat_{mass}', texts)
                   for mass, texts in v2_repeat.items()]
    candidates += [('v2_repeat_0.1_return_0.3', return_texts)]
    candidates += [('v2_varied', varied_texts)]
    candidates += [('v2_varied_low_bracket', varied_low_texts)]
    candidates += [('v8_low_rate', v8_texts),
                   ('alias_low_rate', alias_texts)]
    results = {name: summaries(texts) for name, texts in candidates}
    rule_density = {'v3_priority': dict(v3_labels),
                    'v3_sparse_brackets': dict(v3_sparse_labels),
                    'v2_joint': v2_density}
    rule_density.update({f'v2_subject_repeat_{mass}': density
                         for mass, density in v2_repeat_density.items()})
    rule_density['v2_repeat_0.1_return_0.3'] = return_density
    rule_density['v2_varied'] = varied_density
    rule_density['v2_varied_low_bracket'] = varied_low_density
    panel_note = ('The skip11000 panel is untouched by this audit.'
                  if a.panel == DEFAULT_PANEL else
                  'The skip11000 panel is an independent form check for the '
                  'varied generator, but it was used by earlier alias '
                  'experiments; do not tune on it.')
    output = dict(config={**vars(a), 'panel': str(a.panel),
                          'tokenizer': str(a.tokenizer),
                          'output': str(a.output)},
                  source_sha256=digest, results=results,
                  synthetic_rule_density=rule_density,
                  caveat='Word-pattern proxies, not natural '
                         'task labels. FineWeb and synthetic tokenizers differ. '
                         'The 100-word crop matches surface length only; '
                         'The stopword list is heuristic and raw successor '
                         'copying includes fixed grammatical scaffolds; '
                         'inspect lexical-successor-conditioned copying too. '
                         'v3 square brackets encode quotes, while real square '
                         'brackets can be citations or markup. No source prose '
                         'is exported. '+panel_note)
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    for name, row in results.items():
        print(name, 'repeat/100 content', round(
            row['repeats']['repeat_per_100_content'], 1),
              'copy-rate', round(row['repeats']['same_next_after_repeat_rate'], 3),
              'paren-spans', row['spans']['parenthesis']['complete_spans'])


if __name__ == '__main__':
    main()
