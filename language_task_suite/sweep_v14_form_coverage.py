"""Exploratory V14 layout-rate sweep on a fixed FineWeb development panel."""

import argparse
import hashlib
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from audit_surface_repetition import form_cues, score
from compare_v12_fineweb_fresh import WORD, first_words
from surface_mixture_v6 import render
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from train_v13_attributes import COMMON, REFERENCE


HERE=Path(__file__).resolve().parent
BASE=HERE/'results/v14_conditional_fineweb_skip17000_n64.json'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip',type=int,default=17000)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    ds=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                    split='train',streaming=True)
    rows=list(islice(ds,a.skip,a.skip+64))
    assert len(rows)==64
    hashes=[hashlib.sha256(row['text'].encode()).hexdigest() for row in rows]
    if a.skip==17000:
        baseline=json.loads(BASE.read_text())
        assert hashes==baseline['selected_text_sha256']
    variants={
        'reference_v14':dict(COMMON,**REFERENCE,
                             conditional_continuation_rate=.35),
        'prose_heavy_candidate':dict(
            dict(COMMON,**REFERENCE),
            conditional_continuation_rate=.35,numbered_line_rate=.05,
            quote_rate=.04,source_rate=.08,section_rate=.025,
            scope_question_rate=.04,question_rate=.03,
            price_question_rate=.08),
    }
    generated={}
    toy_seed=99841 if a.skip==17000 else 99851
    for name,kwargs in variants.items():
        gen=SurfaceMixtureV14Conditional(seed=toy_seed,**kwargs)
        texts=[];sites=Counter();streams=Counter()
        for _ in rows:
            ids,labels,_=gen.sample_sequence(256)
            texts.append(render([gen.id2token[i] for i in ids]))
            sites.update(labels)
            streams.update(set(labels))
        generated[name]=dict(texts=texts,sites=dict(sites),
                             streams=dict(streams),kwargs=kwargs)
    natural=[first_words(row['text'],100) for row in rows]
    toy={name:[first_words(v,100) for v in part['texts']]
         for name,part in generated.items()}
    eligible=[i for i in range(64) if len(WORD.findall(natural[i]))>=100
              and all(len(WORD.findall(values[i]))>=100
                      for values in toy.values())]
    assert eligible
    natural=[natural[i] for i in eligible]
    results={}
    for name,part in generated.items():
        texts=[toy[name][i] for i in eligible]
        results[name]=dict(kwargs=part['kwargs'],
                           form=form_cues(texts),
                           repetition=score(texts,40),
                           sites=part['sites'],streams=part['streams'])
    result=dict(source='HuggingFaceFW/fineweb sample-10BT',
                offset=a.skip,toy_seed=toy_seed,selected_text_sha256=hashes,
                selection=('development panel used to choose layout rates'
                           if a.skip==17000 else
                           'fresh held-out offset after layout candidate fixed'),
                n_word_matched=len(eligible),
                natural=dict(form=form_cues(natural),
                             repetition=score(natural,40)),
                variants=results,
                caveat='Exploratory development-panel comparison, not held-out '
                       'validation of naturalness. Form cues do not measure '
                       'semantic dependency realism. No source prose exported.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('matched',len(eligible),'natural',result['natural']['form'],flush=True)
    for name,row in results.items():
        print(name,row['form'],
              {k:row['sites'].get(k,0) for k in
               ('conditional_action','attribute_object','source_answer',
                'scope_answer','price_answer','bracket_close')},flush=True)


if __name__=='__main__':main()
