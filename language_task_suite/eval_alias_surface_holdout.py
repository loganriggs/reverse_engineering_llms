"""Held-out coarse form check after fixing low-rate alias parameters.

Decode text in memory for counts only; export no FineWeb prose or semantics.
The alias rates are fixed from the skip-7000 development panel.
"""

import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from fineweb_random_form_panel import FIRST_TOKENS, surface_stats
from surface_alias_mixture import AliasMixtureLanguage, render


HERE = Path(__file__).resolve().parent
DEV = (HERE.parents[1] / 'reverse_engineer/tensor_language/'
       'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt')
HOLDOUT = (HERE.parents[1] / 'reverse_engineer/tensor_language/'
           'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip11000.pt')
LOW_RATE = dict(parenthetical_rate=.25, definition_start_rate=.10,
                alias_reuse_rate=.10)


def main():
    development = torch.load(DEV, map_location='cpu', weights_only=True)
    holdout = torch.load(HOLDOUT, map_location='cpu', weights_only=True)
    assert development.shape == holdout.shape == (192, 513)
    assert {tuple(row.tolist()) for row in development}.isdisjoint(
        {tuple(row.tolist()) for row in holdout})
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    natural = [tokenizer.decode(row[:FIRST_TOKENS].tolist(),
                                skip_special_tokens=False) for row in holdout]
    gen = AliasMixtureLanguage(seed=20261108, **LOW_RATE)
    synthetic = [render(gen.sample_sequence(FIRST_TOKENS)[0])
                 for _ in range(len(holdout))]
    result = dict(holdout_rows=str(HOLDOUT),
                  holdout_sha256=hashlib.sha256(holdout.numpy().tobytes()).hexdigest(),
                  development_rows=str(DEV),
                  development_sha256=hashlib.sha256(development.numpy().tobytes()).hexdigest(),
                  exact_row_overlap=0, first_gpt2_tokens=FIRST_TOKENS,
                  matched_word_cap=100, generator='AliasMixtureLanguage',
                  generator_seed=20261108, generator_kwargs=LOW_RATE,
                  natural_surface=surface_stats(natural),
                  synthetic_surface=surface_stats(synthetic),
                  caveat='Only coarse form markers, not semantic labels or '
                         'source prose, were inspected on the reserved panel. '
                         'Rows are disjoint by exact tokens; source documents '
                         'may overlap. Tokenizers differ, and a single cue '
                         'frequency is not a natural-language validity test.')
    out = HERE / 'alias_surface_holdout.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print('natural', result['natural_surface'])
    print('alias low rate', result['synthetic_surface'])


if __name__ == '__main__':
    main()
