"""One-annotator, text-independent 16-document review; exports no prose."""

import hashlib
import json
import random
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset


HERE=Path(__file__).resolve().parent
FORM=HERE/'results/v13_attributes_fineweb_skip16000_n64.json'
OUT=HERE/'results/v13_attributes_fineweb_dependency_review_n16.json'
SEED=99822
FIRST_WORDS=160

# An item-attribute description is broader than the V13 rule. A repeated
# attribute requires the same item and property to recur later in the crop.
# The exact synthetic sequence additionally requires a short property fact,
# a later ordinary description, and competing pending item facts; none of
# these 16 crops shows that full pattern.
LABELS={
  3:(('item_attribute','entity_tracking'),
     'A named hat has color and design details across several sentences.'),
  4:(('conditional_action',),
     'A recruiting call to action is conditional on reader interest.'),
  5:(('item_attribute','numeric_binding'),
     'Yarn material, shade count, length and tension stay attached to a product.'),
  18:(('item_attribute','entity_tracking','numeric_binding'),
      'Color combinations and a population count remain attached to a campaign.'),
  19:((),
      'A compilation of mostly unrelated short science-fiction headlines and review text.'),
  25:(('numeric_binding','entity_tracking'),
      'Several distinct spaceflight events and times are linked in a schedule.'),
  27:(('item_attribute','repeated_attribute','numeric_binding','entity_tracking'),
      'A named pear has sweetness and texture stated, then restated later.'),
  34:(('item_attribute','numeric_binding'),
      'A greeting-card listing gives size, paper, envelope and packaging details.'),
  37:(('item_attribute','entity_tracking','list_scope'),
      'An ingredient recurs across prose, episode titles and a recipe list.'),
  38:(('conditional_action','entity_tracking'),
      'Fundraising participation is linked to conditional support and actions.'),
  41:(('conditional_action','scope'),
      'A privacy policy distinguishes permissions, legal requirements and cookie choices.'),
  42:(('item_attribute','numeric_binding','conditional_action'),
      'Furniture descriptions mix materials, finish recommendations and prices.'),
  48:(('numeric_binding','entity_tracking'),
      'A charity and its facilities retain distinct funding amounts and dates.'),
  49:(('numeric_binding','entity_tracking','conditional_action'),
      'A personal medical narrative links dates and a conditional consequence.'),
  52:(('item_attribute','numeric_binding','entity_tracking','scope'),
      'Prints and editions have process, material and numbering attributes.'),
  53:(('attribution','numeric_binding','entity_tracking'),
      'A historical claim is attributed to researchers with time and location context.'),
}


def build():
    form=json.loads(FORM.read_text())
    selected=random.Random(SEED).sample(range(64),16)
    assert set(selected)==set(LABELS)
    ds=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                    split='train',streaming=True)
    rows=list(islice(ds,16000,16064))
    assert len(rows)==64
    word_counts={}
    for i in selected:
        source=rows[i]['text']
        assert hashlib.sha256(source.encode()).hexdigest()==form[
            'selected_text_sha256'][i]
        word_counts[i]=min(len(source.split()),FIRST_WORDS)
    counts=Counter(tag for i in selected for tag in LABELS[i][0])
    return dict(source=FORM.name,selection_seed=SEED,
      selection='16 of 64 fresh-offset documents, uniform without text cues',
      first_words=FIRST_WORDS,
      rows=[dict(index=i,cropped_words=word_counts[i],
                 tags=list(LABELS[i][0]),note=LABELS[i][1])
            for i in selected],tag_counts=dict(counts),
      full_v13_multiple_pending_facts_then_descriptive_reuse=0,
      caveat='One annotator, short crops and overlapping qualitative tags. '
             'A broad relation is not an exact next-token task, and zero '
             'observed full V13 patterns does not establish absence from '
             'FineWeb. No source prose is exported.')


if __name__=='__main__':
    result=build()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(result['tag_counts'],flush=True)
