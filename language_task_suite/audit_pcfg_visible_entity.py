"""Pilot: visible document entity conditions a corpus PCFG proper-name lexicon.

The grammar remains local; this writes only aggregate results. It tests one
state-dependent lexical emission, not compositional semantics or a realistic
document generator.
"""

import argparse
import ast
import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from audit_induced_pcfg_prefix import _read_surprisals
from berkeley_pcfg_sampler import read_berkeley_grammar, sample_sentence
from genre_document_v3 import _base_prefix
from genre_document_v4 import to_v3
from joint_document_v1 import _split, visible_state


def visible_names(prefix):
    canonical = _base_prefix(to_v3(prefix))
    history, raw = _split(canonical)
    assert not raw
    return tuple(visible_state(history)['profiles'])


def boosted_lexicon(source, target, name, strength=.6, floor=.005):
    """Preserve every base emission and add a state-conditioned NNP mixture."""
    lines = source.read_text().splitlines()
    sums = []
    for line in lines:
        tag, _, raw = line.split(maxsplit=2)
        if tag != 'NNP':
            continue
        probs = ast.literal_eval(raw.replace('NaN', '0.0'))
        if not sums:
            sums = [0.]*len(probs)
        assert len(probs) == len(sums)
        for i, p in enumerate(probs):
            sums[i] += p
    assert sums and all(s > 0 for s in sums)
    assert name in ('Alice', 'Bob') and 0 < strength < 1
    floor_weights = [floor*s for s in sums]
    boost_weights = [strength/(1-strength)*(s+2*f)
                     for s, f in zip(sums, floor_weights)]
    def row(word, values):
        return f"NNP {word} [{', '.join(str(v) for v in values)}]"
    target.write_text('\n'.join(lines+[row('Alice', floor_weights),
                                        row('Bob', floor_weights),
                                        row(name, boost_weights)])+'\n')


def _run(command, cwd=None, timeout=120):
    result = subprocess.run(command, cwd=cwd, capture_output=True,
                            text=True, timeout=timeout)
    assert result.returncode == 0, (command, result.stderr[-1000:])


def audit(base, paper_repo, seed=81001):
    profile_a = 'Document : grove . Alice is a pilot .'.split()
    profile_b = 'Document : grove . Bob is a pilot .'.split()
    assert visible_names(profile_a) == ('Alice',)
    assert visible_names(profile_b) == ('Bob',)
    with tempfile.TemporaryDirectory(prefix='pcfg_entity_') as tmp:
        root = Path(tmp)
        variants = {}
        for name in ('Alice', 'Bob'):
            variant = root/name
            shutil.copyfile(base.with_suffix('.grammar'),
                            variant.with_suffix('.grammar'))
            boosted_lexicon(base.with_suffix('.lexicon'),
                            variant.with_suffix('.lexicon'), name)
            nltk_base = root/f'{name}_nltk'
            earley = root/f'{name}_earley.grammar'
            _run(['python3', str(paper_repo/'src/data_generation/'
                                 'process_berkeley_grammar.py'),
                  '-p', str(variant), '-o', str(nltk_base), '-t', '0'])
            _run(['python3', str(paper_repo/'src/lm_eval/'
                                 'process_earleyx_grammar.py'),
                  '-g', str(nltk_base)+'_pcfg.txt',
                  '-o', str(earley), '-s', 'ROOT_0'])
            variants[name] = (variant, earley)
        rules_a, _ = read_berkeley_grammar(variants['Alice'][0])
        rng = random.Random(seed)
        chosen = None
        for attempt in range(5000):
            words, failure = sample_sentence(rules_a, rng)
            if (not failure and 6 <= len(words) <= 25 and
                    words.count('Alice') == 1 and 'Bob' not in words and
                    words.index('Alice') >= 2):
                chosen = words
                break
        assert chosen is not None
        position = chosen.index('Alice')
        rows = [chosen, chosen[:position]+('Bob',)+chosen[position+1:]]
        rows = [tuple(w.replace("'", '<apostrophe>') for w in row)
                for row in rows]
        sentence_file = root/'sentences.txt'
        sentence_file.write_text('\n'.join(' '.join(row) for row in rows)+'\n')
        probabilities = {}
        for name in ('Alice', 'Bob'):
            output = root/f'{name}_score'
            earley = variants[name][1]
            _run(['java', '-Xmx4g', '-classpath',
                  f'{paper_repo}/src/lm_eval/earleyx/earleyx_fast.jar:'
                  f'{paper_repo}/src/lm_eval/earleyx/lib/*',
                  'parser.Main', '-in', str(sentence_file),
                  '-grammar', str(earley), '-out', str(output),
                  '-verbose', '0', '-thread', '1', '-root', 'ROOT_0'],
                 timeout=180)
            scores = _read_surprisals(output.with_suffix('.surprisal'))
            assert len(scores) == 2 and all(len(s) == len(chosen)
                                            for s in scores)
            assert all(math.isfinite(v) for s in scores for _, v in s)
            probabilities[name] = {
                candidate: math.exp(-scores[i][position][1])
                for i, candidate in enumerate(('Alice', 'Bob'))}
    assert probabilities['Alice']['Alice'] > probabilities['Bob']['Alice']
    assert probabilities['Bob']['Bob'] > probabilities['Alice']['Bob']
    return dict(source='paper Berkeley PCFG plus visible-name NNP bias',
                seed=seed, grammar_sha256=hashlib.sha256(
                    base.with_suffix('.grammar').read_bytes()).hexdigest(),
                lexicon_sha256=hashlib.sha256(
                    base.with_suffix('.lexicon').read_bytes()).hexdigest(),
                sampled_sentence_attempts=attempt+1,
                sampled_sentence_tokens=len(chosen),
                target_position=position,
                candidate_probabilities=probabilities,
                state_rule='Visible V4 profile name; NNP lexical prior '
                           'has 0.6 weight on that name and common floor.',
                caveat=('Only a proper-name lexical distribution is changed. '
                        'The sampled sentence may be ungrammatical, and '
                        'these two target probabilities do not sum to one '
                        'because the NNP vocabulary is larger. No source '
                        'fact or relation is conditioned in this PCFG.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grammar-base', type=Path, required=True)
    p.add_argument('--paper-repo', type=Path, required=True)
    p.add_argument('--seed', type=int, default=81001)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.grammar_base, a.paper_repo, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['sampled_sentence_attempts'],
          result['candidate_probabilities'])
