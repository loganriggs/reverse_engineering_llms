"""Diagnostic V10 surface-form comparison on a previously used FineWeb panel."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_surface_repetition import WORD, form_cues, score
from surface_mixture_v6 import render
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents


ROWS=(Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/')/
      'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=Path,default=ROWS)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99101)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=torch.load(a.rows,map_location='cpu',weights_only=True)
    tokenizer=Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    natural=[tokenizer.decode(row[:a.length].tolist(),skip_special_tokens=False)
             for row in rows]
    kwargs=dict(counterbalanced_queries=True,question_rate=.05,count_rate=0.,
                quote_rate=.08,source_rate=.15,section_rate=.06,detail_rate=.8,
                scope_question_rate=.08,source_object_mass=.15,
                source_subject_mass=.15,bracket_rate=.06,
                linked_subject_mass=.65,linked_object_mass=.65,
                linked_question_mass=.30,linked_query_key_mass=.65)
    gen=SurfaceMixtureV10LinkedEvents(seed=a.seed,**kwargs)
    synthetic=[render([gen.id2token[i] for i in gen.sample_sequence(a.length)[0]])
               for _ in range(len(rows))]
    eligible=[i for i,(x,y) in enumerate(zip(natural,synthetic))
              if len(WORD.findall(x))>=40 and len(WORD.findall(y))>=40]
    result=dict(config=dict(rows=str(a.rows),length=a.length,seed=a.seed,
                            n=len(rows),generator_kwargs=kwargs),
                fineweb_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                n_word_matched=len(eligible),
                fineweb=dict(forms=form_cues(natural),repetition=score(natural,40)),
                synthetic=dict(forms=form_cues(synthetic),repetition=score(synthetic,40)),
                caveat='FineWeb panel was used in prior toy-language development; '
                'this is a diagnostic, not a fresh confirmation. A toy token is '
                'not a GPT token, and form/repetition statistics do not measure '
                'semantic dependency prevalence or human naturalness. No FineWeb '
                'prose is exported.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('FineWeb',result['fineweb'])
    print('V10',result['synthetic'])


if __name__=='__main__':main()
