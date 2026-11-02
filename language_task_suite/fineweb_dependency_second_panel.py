"""Reproduce a small, manually coded natural-dependency review.

The 32 windows are selected independently of text cues from the previously
form-only skip-11000 panel. Labels are judgments on the first 160 GPT-2
tokens, not ground-truth next-token distributions. Source prose is not saved.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER


HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/'
              'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip11000.pt')
SOURCE_HASH = '5d6c1697f6d05860e4235c21e6324e3451d47924565d8edb62e06fbe37b3a1fa'
SEED = 20261102
FIRST_TOKENS = 160

# An annotator inspected the sampled excerpts. Tags require an overt local
# relation in this window, not just a word such as "if" or a quote mark.
# The note records the visible evidence without reproducing source prose.
LABELS = {
    145: (('alias_definition', 'scope'), 'SSL is expanded; a notice heading scopes the policy.'),
    16: (('numeric_binding', 'entity_tracking'), 'Years, teams, games and scores are related across sentences.'),
    96: (('attribution', 'question_answer'), 'An interview question is followed by the named respondent.'),
    161: (('attribution', 'alias_definition'), 'A named general is credited with a quotation; COA is expanded.'),
    164: (('scope', 'procedure', 'numeric_binding'), 'A rehearsal subheading scopes instructions and vehicle counts.'),
    40: ((), 'Personal aside; no clear local dependency target in this crop.'),
    31: (('procedure', 'entity_tracking'), 'Orientation requirements and placement tests are explained across sentences.'),
    106: (('alias_definition', 'attribution'), 'A speaker introduces a role and defines a named term.'),
    11: (('scope', 'conditional_action'), 'A two-item list supplies necessary conditions for operating equipment.'),
    166: (('conditional_action', 'entity_tracking'), 'Contingency advice refers back to children and their gadgets.'),
    51: (('alias_definition', 'numeric_binding', 'entity_tracking'), 'Group abbreviations and dated political events recur.'),
    103: (('entity_tracking',), 'A boy and a fictional enforcement team recur in a short scene.'),
    132: (('attribution', 'scope'), 'A testimonial heading attributes a series of quotations to students.'),
    157: (('alias_definition', 'question_answer'), 'YMS is introduced after rhetorical maintenance questions.'),
    119: (('conditional_action',), 'An explicit harassment condition leads to advice about filing a claim.'),
    49: (('attribution', 'scope', 'conditional_action'), 'Numbered verses attribute sayings and include an if-clause consequence.'),
    153: (('numeric_binding', 'entity_tracking'), 'A named team and skipper are linked to match counts and a final.'),
    47: (('attribution', 'scope', 'entity_tracking'), 'Verse numbers and named speakers organize a continuing scene.'),
    143: (('numeric_binding', 'entity_tracking'), 'Distance, elevation, route and the same mountain are linked.'),
    104: (('attribution', 'entity_tracking'), 'A politician is named as the respondent to a quoted reporter question.'),
    144: (('attribution', 'numeric_binding', 'entity_tracking'), 'Named runners and dated wins precede a quoted prediction.'),
    113: (('field_value', 'numeric_binding', 'question_answer'), 'Named test packages bind to prices, counts and a cost question.'),
    151: (('field_value', 'numeric_binding', 'scope'), 'A list binds each named pundit to a separate numerical prediction.'),
    82: (('attribution', 'numeric_binding', 'entity_tracking'), 'A named researcher reports measurements across years.'),
    181: ((), 'Generalized description without a clear locally testable relation.'),
    139: (('question_answer', 'scope', 'conditional_action'), 'A FAQ answer contains a numbered list and conditional advice.'),
    128: (('attribution', 'entity_tracking'), 'Byline and article context bind an author to continued claims.'),
    14: (('field_value', 'numeric_binding', 'conditional_action'), 'Shipping policy binds charges and times to delivery conditions.'),
    162: (('scope', 'procedure', 'conditional_action'), 'A rehearsal heading and contingencies organize instructions.'),
    22: (('field_value', 'numeric_binding', 'scope'), 'Numbered match records bind names, dates and title states.'),
    26: (('numeric_binding', 'entity_tracking'), 'A dated article repeatedly discusses local homes and tours.'),
    18: (('field_value', 'procedure', 'conditional_action'), 'Coupon instructions and product-rating fields are visible.'),
}


def build():
    rows = torch.load(SOURCE, map_location='cpu', weights_only=True)
    assert rows.shape == (192, 513)
    digest = hashlib.sha256(rows.numpy().tobytes()).hexdigest()
    assert digest == SOURCE_HASH
    chosen = random.Random(SEED).sample(range(len(rows)), 32)
    assert set(chosen) == set(LABELS)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    for index in chosen:
        text = tokenizer.decode(rows[index, :FIRST_TOKENS].tolist(),
                                skip_special_tokens=False)
        assert text and len(text) > 100
    counts = Counter(tag for index in chosen for tag in LABELS[index][0])
    return dict(source_rows=str(SOURCE), source_sha256=digest,
                tokenizer=str(DEFAULT_TOKENIZER), selection_seed=SEED,
                first_gpt2_tokens=FIRST_TOKENS,
                selection='uniform without replacement; no text-cue filter',
                label_definitions={
                    'attribution': 'A named or grouped source is explicitly linked to speech, quotation or article claims.',
                    'alias_definition': 'An abbreviation, name or term is locally introduced with its expansion or meaning.',
                    'field_value': 'A structured item or named product is linked to a distinct attribute value.',
                    'numeric_binding': 'A quantity is linked to a particular entity, event, product or time.',
                    'conditional_action': 'A visible condition controls advice, action or consequence.',
                    'scope': 'A heading, numbered entry or list structure groups the following content.',
                    'procedure': 'Steps or requirements organize actions in an order or dependency.',
                    'question_answer': 'A visible question is followed by a locally related answer or respondent.',
                    'entity_tracking': 'The same named entity or specific referent continues across sentences.'},
                rows=[dict(index=i, tags=list(LABELS[i][0]), note=LABELS[i][1])
                      for i in chosen],
                tag_counts=dict(sorted(counts.items())),
                no_clear_relation=sum(not LABELS[i][0] for i in chosen),
                caveat='One annotator, 32 short token windows, overlapping labels, '
                       'not a corpus-wide prevalence estimate. Tags are plausible '
                       'relations, not unique next-token answers or validated '
                       'causal labels. Source prose is not exported. This panel '
                       'was previously used for form-only validation and is now '
                       'development evidence for semantic task design.')


if __name__ == '__main__':
    output = HERE/'fineweb_dependency_second_panel.json'
    result = build()
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['tag_counts'], 'no clear relation', result['no_clear_relation'])
