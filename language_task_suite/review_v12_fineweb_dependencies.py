"""Manual dependency review of a text-independent subsample of fresh FineWeb."""

import hashlib
import json
import random
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset


HERE=Path(__file__).resolve().parent
FORM=HERE/'results/v12_price_faq_fineweb_fresh_skip15000_n64.json'
OUT=HERE/'results/v12_price_faq_fineweb_dependency_review_n16.json'
SEED=99802
FIRST_WORDS=160

# Judgments on the first 160 words at each selected offset. A tag requires a
# visible relation, not a matching punctuation mark. The exact V12 triplet
# requires an item+number fact, intervening mention, and a later cost question
# about that same item in the crop; none of these 16 crops met that bar.
LABELS={
  1:(('entity_tracking','numeric_binding'),
     'School, year and travel narrative remain associated.'),
  4:(('field_value','numeric_binding','entity_tracking'),
     'Film title, release date, director and cast are linked.'),
  8:(('field_value','numeric_binding','entity_tracking'),
     'Product pack is described with template count and print resolution.'),
  11:(('attribution','entity_tracking'),
      'Named governor is linked to a quoted economic claim and critique.'),
  13:(('question_answer','attribution'),
      'Rhetorical questions are answered in a discussion of a cited prayer.'),
  14:(('attribution','entity_tracking'),
      'Named governor is attached to a continuing quotation.'),
  16:(('conditional_action','entity_tracking'),
      'Legal advice depends on whether an attorney-client relationship exists.'),
  19:(('numeric_binding','entity_tracking'),
      'Two named senators have distinct meeting counts and intervals.'),
  20:(('field_value','entity_tracking'),
      'A named sports kit carries colors and design details across sentences.'),
  24:(('numeric_binding','entity_tracking','attribution'),
      'Named charger is linked to an order quantity and a company statement.'),
  25:(('field_value','entity_tracking'),
      'A forum listing links a vehicle to an asking price and mileage.'),
  33:(('field_value','numeric_binding'),
      'Named publication is linked to a count of designs.'),
  35:(('field_value','numeric_binding','scope'),
      'Hotel listing groups room types, address and nearby distances.'),
  39:(('entity_tracking',),
      'A personal story continues about a giveaway and family prank.'),
  46:(('entity_tracking',),
      'Travel and training plans continue with the same narrator.'),
  47:(('field_value','numeric_binding','conditional_action','scope'),
      'Room listing groups property type, nearby distances and house rules.'),
}


def build():
    form=json.loads(FORM.read_text())
    selected=random.Random(SEED).sample(range(64),16)
    assert set(selected)==set(LABELS)
    dataset=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                         split='train',streaming=True)
    rows=list(islice(dataset,15000,15064))
    assert len(rows)==64
    word_counts={}
    for i in selected:
        text=rows[i]['text']
        assert hashlib.sha256(text.encode()).hexdigest()==form[
            'selected_text_sha256'][i]
        word_counts[i]=min(len(text.split()),FIRST_WORDS)
    counts=Counter(tag for i in selected for tag in LABELS[i][0])
    return dict(source=FORM.name,selection_seed=SEED,
      selection='16 of 64 contiguous fresh-offset documents, uniformly '
                'sampled without text-cue filtering',
      first_words=FIRST_WORDS,
      rows=[dict(index=i,cropped_words=word_counts[i],
                 tags=list(LABELS[i][0]),note=LABELS[i][1])
            for i in selected],tag_counts=dict(counts),
      exact_v12_item_price_prose_cost_question_triplets=0,
      caveat='One annotator, 16 short crops. Tags are qualitative overlapping '
             'relations, not exact next-token targets. Zero observed exact '
             'triplets does not prove none exist in FineWeb. The broader '
             'item-attribute and numeric-binding relations are present, but '
             'the V12 template is a synthetic stress test rather than a '
             'natural-frequency model. No source prose is exported.')


if __name__=='__main__':
    result=build()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(result['tag_counts'],'exact V12 triplets',
          result['exact_v12_item_price_prose_cost_question_triplets'])
