"""Extend the text-independent V13 FineWeb review from 16 to 32 crops."""

import hashlib
import json
import random
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from review_v13_fineweb_dependencies import (
    FIRST_WORDS, FORM, HERE, LABELS as FIRST_LABELS,
)


OUT=HERE/'results/v13_attributes_fineweb_dependency_review_n32.json'
SECOND_SEED=99823
SECOND_LABELS={
  0:(('item_attribute','repeated_attribute','numeric_binding'),
     'A named buoyancy aid has size repeated between the title and specifications.'),
  1:(('attribution','numeric_binding','entity_tracking'),
     'A casting report keeps actors, roles, dates and an agency source distinct.'),
  2:(('conditional_action','entity_tracking'),
     'Advice about arguing has condition-dependent consequences across prose.'),
  6:(('numeric_binding','entity_tracking'),
     'A research abstract links disease severity to several named brain regions.'),
  10:(('item_attribute','numeric_binding'),
      'A glasses case listing links color and dimensions to the item.'),
  12:(('item_attribute','question_answer','conditional_action'),
      'A machinery FAQ includes cost and configuration questions with conditional answers.'),
  21:(('numeric_binding','entity_tracking','attribution'),
      'A webinar article binds speakers, organizations, date and meeting roles.'),
  22:(('numeric_binding','entity_tracking'),
      'Two named sponsors are linked to different contributions and dates.'),
  35:(('conditional_action','numeric_binding'),
      'A business pitch conditions follow-up advice and cites a percentage claim.'),
  36:(('item_attribute','repeated_attribute','entity_tracking'),
      'Rich texture of a named body cream is restated across adjacent prose.'),
  46:(('item_attribute','repeated_attribute','entity_tracking'),
      'A named onyx sculpture and the stone properties recur in later prose.'),
  54:(('entity_tracking',),
      'A short philosophical synopsis remains about the same thinker.'),
  56:(('entity_tracking',),
      'A personal narrative tracks several people and travel plans.'),
  58:(('conditional_action','scope','list_scope'),
      'A device instruction scopes steps to products and selected input settings.'),
  62:(('conditional_action','list_scope'),
      'A political critique gives conditional writing advice in a list.'),
  63:(('attribution','numeric_binding','entity_tracking'),
      'An artist interview links quotations, artworks and years.'),
}


def build():
    form=json.loads(FORM.read_text())
    first=random.Random(99822).sample(range(64),16)
    second=random.Random(SECOND_SEED).sample(
        [i for i in range(64) if i not in first],16)
    assert set(first)==set(FIRST_LABELS)
    assert set(second)==set(SECOND_LABELS)
    labels=dict(FIRST_LABELS)
    labels.update(SECOND_LABELS)
    ds=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                    split='train',streaming=True)
    rows=list(islice(ds,16000,16064))
    assert len(rows)==64
    word_counts={}
    for i in first+second:
        source=rows[i]['text']
        assert hashlib.sha256(source.encode()).hexdigest()==form[
            'selected_text_sha256'][i]
        word_counts[i]=min(len(source.split()),FIRST_WORDS)
    counts=Counter(tag for i in first+second for tag in labels[i][0])
    return dict(source=FORM.name,
      selection='Two sequential 16-document uniform draws without text cues '
                'from the same 64-document fresh-offset frame',
      selection_seeds=[99822,SECOND_SEED],first_words=FIRST_WORDS,
      rows=[dict(index=i,cropped_words=word_counts[i],
                 tags=list(labels[i][0]),note=labels[i][1])
            for i in first+second],tag_counts=dict(counts),
      full_v13_multiple_pending_facts_then_descriptive_reuse=0,
      caveat='One annotator, 32 short crops and overlapping qualitative tags. '
             'A broad relation is not an exact next-token task. This is a '
             'larger follow-up to the initial 16, not an independent corpus. '
             'Zero observed full V13 patterns does not establish absence '
             'from FineWeb. No source prose is exported.')


if __name__=='__main__':
    result=build()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(result['tag_counts'],flush=True)
