"""Diagnostic V10/V11 form comparison on a previously used FineWeb panel."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_surface_repetition import WORD,form_cues,score
from audit_v11_section_topic import REFERENCE
from surface_mixture_v6 import render
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from surface_mixture_v11_section_topic import SurfaceMixtureV11SectionTopic


ROWS=(Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/')/
      'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=Path,default=ROWS)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=98601)
    p.add_argument('--section-rate',type=float,default=REFERENCE['section_rate'])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=torch.load(a.rows,map_location='cpu',weights_only=True)
    tokenizer=Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    natural=[tokenizer.decode(row[:a.length].tolist(),skip_special_tokens=False)
             for row in rows]
    kwargs=dict(REFERENCE,section_rate=a.section_rate)
    generated={}
    for name,cls,extra in [('v10',SurfaceMixtureV10LinkedEvents,{}),
                           ('v11',SurfaceMixtureV11SectionTopic,
                            {'section_topic_mass':.35})]:
        gen=cls(seed=a.seed,**kwargs,**extra)
        generated[name]=[render([gen.id2token[i] for i in
                                 gen.sample_sequence(a.length)[0]])
                         for _ in range(len(rows))]
    eligible=[i for i in range(len(rows))
              if all(len(WORD.findall(texts[i]))>=40
                     for texts in (natural,*generated.values()))]
    assert eligible
    def stats(texts):
        selected=[texts[i] for i in eligible]
        return dict(forms=form_cues(selected),repetition=score(selected,40))
    output=dict(config=dict(rows=str(a.rows),seed=a.seed,length=a.length,
                            generator_kwargs=kwargs,n=len(rows)),
                source_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                n_word_matched=len(eligible),fineweb=stats(natural),
                v10=stats(generated['v10']),v11=stats(generated['v11']),
                caveat='This FineWeb panel has been used in earlier development. '
                       'These are diagnostic punctuation/repetition metrics, '
                       'not semantic dependency prevalence. V10 and V11 have '
                       'the same surface grammar and may differ only through '
                       'sampled word choices. No FineWeb prose is exported.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print('n',len(eligible))
    for name in ('fineweb','v10','v11'):
        print(name,output[name])


if __name__=='__main__':main()
