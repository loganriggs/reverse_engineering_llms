"""Diagnostic form check of v3 bracket successor against disjoint FineWeb rows."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from generator_v3 import LanguageV3
from original_v3_shared_bracket import LanguageV3SharedBracket
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_surface_repetition import WORD,form_cues,score
from fineweb_random_form_panel import first_words


ROWS=(Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/')/
      'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt')


def synthetic(gen,n,length):
    return [' '.join(gen.id2token[i] for i in gen.sample_sequence(length)[0])
            for _ in range(n)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,
                   default=Path('/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'))
    p.add_argument('--rows',type=Path,default=ROWS)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--seed',type=int,default=97201)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=torch.load(a.rows,map_location='cpu',weights_only=True)
    tokenizer=Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    natural=[tokenizer.decode(row[:a.length].tolist(),skip_special_tokens=False)
             for row in rows]
    old=synthetic(LanguageV3(seed=a.seed),len(rows),a.length)
    new=synthetic(LanguageV3SharedBracket(seed=a.seed),len(rows),a.length)
    eligible=[i for i in range(len(rows)) if all(len(WORD.findall(group[i]))>=a.words
             for group in (natural,old,new))]
    assert eligible
    def summary(group):
        cropped=[first_words(group[i],a.words) for i in eligible]
        return dict(forms=form_cues(cropped),repetition=score(cropped,a.words))
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                fineweb_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                n_word_matched=len(eligible),fineweb=summary(natural),
                original_v3=summary(old),shared_bracket_v3=summary(new),
                caveat='This disjoint FineWeb panel has been used in earlier '
                       'toy-language development, so the comparison is '
                       'diagnostic, not fresh confirmation. GPT and toy tokens '
                       'differ. Form/repetition statistics cannot establish '
                       'semantic task realism. No FineWeb prose is exported.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('n',len(eligible))
    for name in ('fineweb','original_v3','shared_bracket_v3'):
        print(name,result[name])


if __name__=='__main__':main()
