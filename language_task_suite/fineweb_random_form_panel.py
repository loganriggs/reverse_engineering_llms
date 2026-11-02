"""Reproduce a random FineWeb form audit; export IDs and labels, never prose.

The 32 windows are sampled independently of regex matches from the disjoint
skip-7000 development panel, excluding its previously cue-selected windows.
One annotator inspected the first 160 GPT-2 tokens of each selected window.
The labels describe a conspicuous local relation, not an exact LM target.
"""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from surface_mixture_v2 import SurfaceMixtureV2, render
from surface_mixture_v7 import render as render_v8
from surface_mixture_v8 import SurfaceMixtureV8
from surface_alias_mixture import AliasMixtureLanguage, render as render_alias


HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'fineweb_dependency_panel_skip7000.json'
SEED = 20261104
FIRST_TOKENS = 160
MATCHED_WORDS = 100

# Form labels: article, essay, personal, fiction, structured, instructional.
# Relation labels name only the most conspicuous possible toy task. They are
# deliberately qualitative and not a claim of a unique next-token answer.
LABELS = {
    181: ('structured', 'field_binding', 'Profile fields link schools and admissions.'),
    62: ('personal', 'conditional', 'Religious exhortation includes an explicit if-clause.'),
    94: ('article', 'attribution', 'A review quotation and named contributors appear in a digest.'),
    106: ('article', 'none_clear', 'The excerpt explains disease risks without one clean local target.'),
    141: ('article', 'attribution', 'A named colleague is linked to direct quotations.'),
    170: ('article', 'numeric_fact', 'Teams, scores, rankings and dates are bound in a sports report.'),
    5: ('article', 'attribution', 'A named official is linked to several direct quotations.'),
    67: ('instructional', 'procedure', 'Measurements and repeated actions form a procedure.'),
    111: ('article', 'alias', 'An acronym is introduced beside its expansion.'),
    1: ('essay', 'coreference', 'Pronouns and descriptions refer back to named people.'),
    155: ('structured', 'attribution', 'Meeting-minute initials attribute statements and plans.'),
    168: ('essay', 'coreference', 'Groups and leaders are referred to across sentences.'),
    19: ('structured', 'field_binding', 'Publication title and authors occupy distinct record fields.'),
    131: ('instructional', 'numeric_fact', 'A small-business cybersecurity statistic is stated.'),
    108: ('article', 'numeric_fact', 'Loan amounts and beneficiary counts attach to programs.'),
    7: ('essay', 'conditional', 'Obedience and non-obedience have contrasted consequences.'),
    127: ('structured', 'field_binding', 'Product metadata and age guidance are grouped.'),
    48: ('essay', 'local_repetition', 'The article title is repeated at the start of the body.'),
    157: ('structured', 'attribution', 'Meeting-minute initials attribute an organization update.'),
    134: ('essay', 'none_clear', 'Speech transitions among related cybersecurity topics.'),
    71: ('structured', 'field_binding', 'Fiction metadata lists author, rating, status and story.'),
    31: ('essay', 'question_answer', 'A theological question is followed by an argued answer.'),
    83: ('fiction', 'attribution', 'Dialogue turns are associated with characters.'),
    84: ('fiction', 'coreference', 'Pronouns track participants in a multi-step scene.'),
    63: ('personal', 'none_clear', 'The excerpt is a personal argument with no clean local task.'),
    30: ('essay', 'conditional', 'A counterfactual about a person motivates a later argument.'),
    92: ('article', 'none_clear', 'A comics blog references artists without a clean local target.'),
    177: ('fiction', 'attribution', 'Dialogue turns and a conditional plan involve several speakers.'),
    142: ('structured', 'section_scope', 'Question and bullet answers create distinct item scopes.'),
    186: ('instructional', 'section_scope', 'Recipe headings scope their following descriptions.'),
    150: ('personal', 'attribution', 'A quoted verse and a pastor are locally associated.'),
    140: ('structured', 'section_scope', 'Navigation items are grouped under a section heading.'),
}


def first_words(text, n=MATCHED_WORDS):
    matches = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    return text[:matches[n-1].end()] if len(matches) >= n else text


def surface_stats(texts):
    cropped = [first_words(text) for text in texts]
    return dict(n_windows=len(cropped),
                mean_words=round(sum(len(re.findall(r'[A-Za-z]+', s))
                                     for s in cropped)/len(cropped), 1),
                windows_with_newline=sum('\n' in s for s in cropped),
                windows_with_question=sum('?' in s for s in cropped),
                windows_with_colon=sum(':' in s for s in cropped),
                windows_with_quote=sum(any(mark in s for mark in ('"', '“', '”'))
                                       for s in cropped),
                windows_with_parenthesis=sum('(' in s or ')' in s
                                             for s in cropped),
                windows_with_digit=sum(bool(re.search(r'\d', s)) for s in cropped))


def build():
    old = json.loads(SOURCE.read_text())
    rows_path = Path(old['rows'])
    rows = torch.load(rows_path, map_location='cpu', weights_only=True)
    assert rows.shape == (192, 513)
    source_hash = hashlib.sha256(rows.numpy().tobytes()).hexdigest()
    assert source_hash == old['source_sha256']
    previous = {entry['index'] for group in old['groups'].values()
                for entry in group['rows']}
    chosen = random.Random(SEED).sample(
        [i for i in range(len(rows)) if i not in previous], 32)
    assert set(chosen) == set(LABELS)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    natural = [tokenizer.decode(rows[i, :FIRST_TOKENS].tolist(),
                                skip_special_tokens=False) for i in chosen]
    generator = SurfaceMixtureV2(seed=SEED)
    synthetic = [render(generator.sample_sequence(FIRST_TOKENS)[0])
                 for _ in chosen]
    v8 = SurfaceMixtureV8(seed=SEED, section_rate=.06, detail_rate=.8,
                          scope_question_rate=.08, quote_rate=.08,
                          source_rate=.15)
    synthetic_v8 = [render_v8(v8.sample_sequence(FIRST_TOKENS)[0])
                    for _ in chosen]
    alias_generator = AliasMixtureLanguage(seed=SEED)
    synthetic_alias = [render_alias(alias_generator.sample_sequence(FIRST_TOKENS)[0])
                       for _ in chosen]
    alias_low = AliasMixtureLanguage(seed=SEED, parenthetical_rate=.25,
                                     definition_start_rate=.10,
                                     alias_reuse_rate=.10)
    synthetic_alias_low = [render_alias(alias_low.sample_sequence(FIRST_TOKENS)[0])
                           for _ in chosen]
    entries = [dict(index=i, form=LABELS[i][0], relation=LABELS[i][1],
                    reason=LABELS[i][2]) for i in chosen]
    return dict(source_rows=str(rows_path), source_sha256=source_hash,
                selection_seed=SEED, first_gpt2_tokens=FIRST_TOKENS,
                matched_word_cap=MATCHED_WORDS,
                excluded_prior_cue_rows=sorted(previous), rows=entries,
                form_counts=dict(Counter(item['form'] for item in entries)),
                relation_counts=dict(Counter(item['relation'] for item in entries)),
                natural_surface=surface_stats(natural),
                synthetic_v2_surface=surface_stats(synthetic),
                synthetic_v8_low_rate_surface=surface_stats(synthetic_v8),
                synthetic_alias_task_rich_surface=surface_stats(synthetic_alias),
                synthetic_alias_low_rate_surface=surface_stats(synthetic_alias_low),
                caveat='One annotator, 32 random windows from a previously used '
                       'development panel after cue-row exclusions. Relation '
                       'labels are hypotheses from short excerpts, not exact '
                       'next-token targets or FineWeb prevalence estimates. '
                       'Natural and toy tokenizers differ; compare only broad '
                       'form, with both texts cropped to at most 100 words. '
                       'No source prose is stored.')


if __name__ == '__main__':
    result = build()
    output = HERE / 'fineweb_random_form_panel.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('forms', result['form_counts'])
    print('relations', result['relation_counts'])
    print('natural', result['natural_surface'])
    print('toy v2', result['synthetic_v2_surface'])
    print('toy v8 low rate', result['synthetic_v8_low_rate_surface'])
    print('toy alias task rich', result['synthetic_alias_task_rich_surface'])
    print('toy alias low rate', result['synthetic_alias_low_rate_surface'])
