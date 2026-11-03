"""Audit exact-prefix support and source/narrative overlap in the v9 pilot."""

import argparse
import json
from collections import Counter
from pathlib import Path

from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v8 import SurfaceMixtureV8
from surface_mixture_v9_source_bridge import SurfaceMixtureV9SourceBridge


def panel(n, length, seed, kwargs):
    gen = SurfaceMixtureV9SourceBridge(seed=seed, **kwargs)
    control = SurfaceMixtureV8(seed=seed, **{k:v for k,v in kwargs.items()
                                           if k not in ('source_object_mass',
                                                        'source_subject_mass',
                                                        'bracket_rate')})
    labels = Counter()
    streams = Counter()
    source_object_sites = 0
    source_object_collisions = 0
    source_object_target_masses = []
    source_object_examples = []
    source_answer_sites = 0
    source_answer_and_object_streams = 0
    bracket_streams = 0
    all_three_streams = 0
    bracket_swap_checks = 0
    for i in range(n):
        ids, labs, qs = gen.sample_sequence(length)
        words = [gen.id2token[t] for t in ids]
        assert len(words)==len(labs)==len(qs)==length
        labels.update(labs)
        had_source_object = False
        for t,(word,q) in enumerate(zip(words,qs)):
            assert abs(sum(q.values())-1)<1e-9 and q.get(word,0)>0
            assert q == gen.next_distribution(words[:t])
            if labs[t]=='source_answer':
                source_answer_sites += 1
                assert q == {word:1.}
            if labs[t]=='bracket_close':
                opener = '(' if ')' in q else '['
                closer = ')' if opener == '(' else ']'
                assert closer in q and q[closer] > 0
                assert q.get(')' if opener == '[' else ']',0) == 0
                k = max(k for k in range(t) if words[k] == opener and
                        '.' not in words[k:t])
                edited = words[:t].copy()
                edited[k] = '[' if opener == '(' else '('
                swapped = gen.next_distribution(edited)
                other = ']' if closer == ')' else ')'
                assert abs(swapped[other]-q[closer]) < 1e-10
                assert {z:p for z,p in swapped.items() if z != other} == {
                    z:p for z,p in q.items() if z != closer}
                bracket_swap_checks += 1
            source = gen.source_object_candidates(words[:t])
            if not source:
                continue
            had_source_object = True
            source_object_sites += 1
            assert labs[t]=='mixed_object'
            base = control.next_distribution(words[:t])
            source_dist = {obj:1/len(source) for obj in source}
            expected = {token:(1-gen.source_object_mass)*p for token,p in base.items()}
            for obj,p in source_dist.items():
                expected[obj] = expected.get(obj,0.)+gen.source_object_mass*p
            assert set(q)==set(expected)
            assert max(abs(q[k]-v) for k,v in expected.items())<1e-10
            base_candidates = gen.object_candidates(words[:t])
            overlap = set(source)&set(base)
            source_object_collisions += bool(overlap)
            source_object_target_masses.append(sum(q.get(k,0) for k in source))
            if len(source_object_examples)<8:
                history, raw = _split(words[:t])
                content = _content(raw)
                frame = _frame(content)
                source_object_examples.append(dict(prefix=' '.join(words[max(0,t-40):t]),
                    subject=content[_indices(frame)[0]], source_objects=list(source),
                    base_candidates=base_candidates,
                    base_probability_on_source=sum(base.get(k,0) for k in source),
                    source_component_mass=gen.source_object_mass,
                    source_token_probability=sum(q.get(k,0) for k in source),
                    sampled=word, overlapping_outputs=sorted(overlap)))
        present=set(labs)
        streams.update(present)
        source_answer_and_object_streams += had_source_object and 'source_answer' in present
        bracket_streams += 'bracket_close' in present
        all_three_streams += (had_source_object and 'source_answer' in present
                              and 'bracket_close' in present)
    return dict(config=dict(n=n,length=length,seed=seed,kwargs=kwargs),
                labels=dict(labels),streams_with_label=dict(streams),
                source_object_sites=source_object_sites,
                source_object_collision_sites=source_object_collisions,
                source_answer_sites=source_answer_sites,
                streams_with_source_answer_and_object=source_answer_and_object_streams,
                streams_with_bracket=bracket_streams,
                streams_with_source_answer_object_and_bracket=all_three_streams,
                bracket_opener_swap_checks=bracket_swap_checks,
                source_object_target_probability_mean=(
                    sum(source_object_target_masses)/len(source_object_target_masses)
                    if source_object_target_masses else None),
                examples=source_object_examples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=100)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=96001)
    p.add_argument('--regime',choices=('diagnostic','reference_candidate'),
                   default='diagnostic')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    # Diagnostic rates deliberately favor overlap; these are not reference
    # estimates of language-task prevalence.
    if a.regime == 'diagnostic':
        kwargs=dict(counterbalanced_queries=True,question_rate=.05,
                    count_rate=0.,quote_rate=.30,source_rate=.30,
                    section_rate=.20,detail_rate=.8,scope_question_rate=.25,
                    source_object_mass=.25,source_subject_mass=.25,
                    bracket_rate=.12)
    else:
        kwargs=dict(counterbalanced_queries=True,question_rate=.05,
                    count_rate=0.,quote_rate=.08,source_rate=.15,
                    section_rate=.06,detail_rate=.8,scope_question_rate=.08,
                    source_object_mass=.15,source_subject_mass=.15,
                    bracket_rate=.06)
    rows=[panel(a.n,a.length,a.seed+j,kwargs) for j in (0,1)]
    output=dict(config=vars(a),panels=rows,caveat=(
        'This is a source-to-narrative overlap pilot. The reference_candidate '
        'rates are provisional, not FineWeb-calibrated. Brackets are in the '
        'same narrative stream; section-to-narrative sharing remains.'))
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n')
    print([(r['source_object_sites'],r['source_object_collision_sites'],
            r['source_answer_sites'],r['streams_with_source_answer_and_object'],
            r['labels'].get('bracket_close',0),
            r['streams_with_source_answer_object_and_bracket'])
           for r in rows])


if __name__=='__main__':main()
