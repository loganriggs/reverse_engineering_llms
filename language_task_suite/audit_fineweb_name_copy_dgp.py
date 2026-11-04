"""Audit eligible corpus carriers and generated name-copy interventions.

Reads an external local preview JSON; writes aggregate results to the repo
and optional source excerpts to a local review file only.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
import statistics

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from fineweb_name_copy_dgp import NameCopyDGP, WORD


def _window(text, point, before=120, after=90):
    return text[max(0,point-before):min(len(text),point+after)].replace('\n',' ')


def audit(rows, seed=45201, review_n=12, mode='full_person',
          max_form_docs=256):
    dgp = NameCopyDGP(rows, mode=mode)
    rng = random.Random(seed)
    generated, original, review = [], [], []
    spans_per_doc, distances, choices = [], [], []
    within_100 = within_250 = 0
    for index in dgp.eligible:
        source = rng.choice(tuple(dgp.candidates[index]))
        replacement = rng.choice(dgp.replacements(index, source))
        sample = dgp.make(index, source, replacement)
        edited = dgp.counterfactual(sample, rng)
        assert edited.original == sample.original and \
            edited.replacement != sample.replacement and \
            len(edited.sites) == len(sample.sites)
        assert all(sample.text[a:b] == replacement for a,b in sample.sites)
        assert all(edited.text[a:b] == edited.replacement
                   for a,b in edited.sites)
        text = rows[index]['preview']
        for item in (sample, edited):
            recovered = re.sub(r'(?<!\w)'+re.escape(item.replacement)+
                               r'(?!\w)', item.original, item.text)
            if recovered != text:
                difference = next((j for j, (a,b) in enumerate(
                    zip(recovered, text)) if a != b),
                    min(len(recovered), len(text)))
                raise AssertionError(
                    f'Round-trip failed at source offset '
                    f"{rows[index]['index']}, original={source}, "
                    f'replacement={item.replacement}, char={difference}')
        original.append(first_words(text, 100))
        generated.append(first_words(sample.text, 100))
        first = dgp.candidates[index][source][0]
        first_target = dgp.candidates[index][source][1]
        distance = len(WORD.findall(text[first:first_target]))
        distances.append(distance)
        spans_per_doc.append(len(sample.sites))
        choices.append(sample.replacement_count)
        target_word_index = len(WORD.findall(text[:first_target]))
        within_100 += target_word_index < 100
        within_250 += target_word_index < 250
        if len(review) < review_n:
            review.append(dict(source_index=sample.source_index,
                               source=source, replacement=replacement,
                               alternate=edited.replacement,
                               first_mention=_window(sample.text, first),
                               first_target=_window(sample.text,
                                                    sample.sites[0][0])))
    all_natural = [first_words(row['preview'], 100) for row in rows]
    form_rng = random.Random(seed+99)
    eligible_form_indices = sorted(form_rng.sample(
        range(len(original)), min(len(original), max_form_docs)))
    all_form_indices = sorted(form_rng.sample(
        range(len(all_natural)), min(len(all_natural), max_form_docs)))
    def summary(panel):
        eligible = [text for text in panel if len(WORD.findall(text)) >= 100]
        return structure(eligible) if eligible else None
    import spacy
    import en_core_web_sm
    result = dict(source='HuggingFaceFW/fineweb sample-10BT train',
                  mode=mode,
                  dependencies={'spacy':spacy.__version__,
                                'en_core_web_sm':en_core_web_sm.__version__,
                                'faker':version('faker')},
                  panel_offsets=[rows[0]['index'], rows[-1]['index']],
                  seed=seed, documents=len(rows),
                  eligible_documents=len(dgp.eligible),
                  eligible_fraction=len(dgp.eligible)/len(rows),
                  replacement_pool_size=len(dgp.pool),
                  target_sites=sum(spans_per_doc),
                  sites_per_eligible_mean=statistics.mean(spans_per_doc),
                  first_target_distance_words_median=statistics.median(distances),
                  first_target_within_first_100_words=within_100,
                  first_target_within_first_250_words=within_250,
                  replacement_choices_min=min(choices),
                  replacement_choices_median=statistics.median(choices),
                  form_panel_cap=max_form_docs,
                  all_natural_form=summary([all_natural[i]
                                            for i in all_form_indices]),
                  eligible_original_form=summary([original[i]
                                                  for i in eligible_form_indices]),
                  eligible_modified_form=summary([generated[i]
                                                  for i in eligible_form_indices]),
                  source_sha256_by_offset=[
                      dict(offset=rows[i]['index'],
                           hash=hashlib.sha256(rows[i]['preview'].encode()).hexdigest())
                      for i in dgp.eligible],
                  caveat='Automatic name selection is fallible even with '
                         'PERSON tags; manual review of edited text is '
                         'required. Substitution can break meaning. '
                         'This is a corpus-conditioned, not population, DGP; '
                         'it has no exact all-token oracle. Source text '
                         'and local manual review are not stored in Git.')
    return result, review


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--review-out', type=Path)
    p.add_argument('--review-n', type=int, default=12)
    p.add_argument('--max-form-docs', type=int, default=256)
    p.add_argument('--mode', choices=('full_person','broad_regex'),
                   default='full_person')
    a = p.parse_args()
    rows = json.loads(a.preview_json.read_text())
    result, review = audit(rows, mode=a.mode, review_n=a.review_n,
                           max_form_docs=a.max_form_docs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    if a.review_out:
        a.review_out.parent.mkdir(parents=True, exist_ok=True)
        a.review_out.write_text(json.dumps(review, indent=2)+'\n')
    print({k:v for k,v in result.items() if not k.endswith('_form') and
           k != 'source_sha256_by_offset'},flush=True)


if __name__ == '__main__':
    main()
