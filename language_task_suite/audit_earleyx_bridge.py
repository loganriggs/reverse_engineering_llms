"""Check the paper's EarleyX interface against an enumerated ambiguous CFG."""

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

from audit_genre_document_v7 import check_exhaustive


TOY_GRAMMAR = """ROOT->[S] : 1.0
S->[A DOT] : 1.0
A->[X Y] : 0.4
A->[X Z] : 0.6
X->[_a] : 0.7
X->[_b] : 0.3
Y->[_c] : 0.5
Y->[_d] : 0.5
Z->[_c] : 0.25
Z->[_e] : 0.75
DOT->[_.] : 1.0
"""


def audit(jar, lib_dir):
    check_exhaustive()
    with tempfile.TemporaryDirectory(prefix='earleyx_oracle_') as tmp:
        root = Path(tmp)
        grammar = root/'toy.grammar'
        text = root/'toy.txt'
        output = root/'score'
        grammar.write_text(TOY_GRAMMAR)
        text.write_text('a c .\n')
        command = ['java', '-Xmx1g', '-classpath',
                   f'{jar}:{lib_dir}/*', 'parser.Main',
                   '-in', str(text), '-grammar', str(grammar),
                   '-out', str(output), '-verbose', '0',
                   '-thread', '1', '-root', 'ROOT']
        process = subprocess.run(command, capture_output=True, text=True,
                                 check=True, timeout=60)
        lines = (root/'score.surprisal').read_text().splitlines()
        observed = [(parts[0], float(parts[1])) for line in lines
                    if len(parts := line.split()) == 2 and
                    not line.startswith('#')]
        expected = [('a', -math.log(.7)),
                    ('c', -math.log(.4*.5+.6*.25)),
                    ('.', 0.)]
        assert [w for w, _ in observed] == [w for w, _ in expected]
        max_error = max(abs(got-want)
                        for (_, got), (_, want) in zip(observed, expected))
        assert max_error < 1e-9, (observed, expected, process.stdout)
        return dict(checked_sentence='a c .',
                    expected_surprisal_nats=[v for _, v in expected],
                    observed_surprisal_nats=[v for _, v in observed],
                    maximum_absolute_error=max_error,
                    exhaustive_checked_prefixes=21)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--jar', type=Path, required=True)
    p.add_argument('--lib-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.jar, args.lib_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result)
