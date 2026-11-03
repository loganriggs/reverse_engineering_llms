"""Audit event ordering, exact oracle support, and linked-rule interventions."""

import argparse
import json
from collections import Counter
from pathlib import Path

from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v10_linked_events import (SurfaceMixtureV10LinkedEvents,
                                                linked_state)
from surface_mixture_v9_source_bridge import SurfaceMixtureV9SourceBridge


def completed_event_triples(words):
    history, _ = _split(words)
    claims, prose, queried = {}, {}, set()
    completed = set()
    distances = []
    for clause_i,raw in enumerate(history):
        clause = _content(raw)
        if not clause:
            continue
        if clause[0]=='As':
            obj = clause[6] if clause[4]=='"' else clause[5]
            claims[obj] = (clause[1],clause_i)
        elif clause[0]=='Who':
            obj = clause[3]
            queried.add(obj)
            if obj in prose:
                completed.add(obj)
                distances.append((prose[obj][1]-claims[obj][1]-1,
                                  clause_i-prose[obj][1]-1))
        elif clause[0] not in ('Where','How','Section','Detail','What'):
            frame = _frame(clause)
            subject, obj = clause[_indices(frame)[0]],clause[OBJECT_INDEX[frame]]
            if obj in claims and claims[obj][0]==subject:
                prose[obj]=(subject,clause_i)
    return dict(claims=len(claims),linked=len(prose),queried=len(queried),
                complete=len(distances),complete_objects=sorted(completed),
                complete_distractor_clauses=distances)


def panel(n,length,seed,kwargs):
    gen=SurfaceMixtureV10LinkedEvents(seed=seed,**kwargs)
    labels=Counter()
    metrics=Counter()
    samples=[]
    for stream in range(n):
        ids,labs,qs=gen.sample_sequence(length)
        words=[gen.id2token[t] for t in ids]
        assert len(ids)==len(labs)==len(qs)==length
        labels.update(labs)
        events=completed_event_triples(words)
        metrics['streams_with_claim'] += events['claims']>0
        metrics['streams_with_linked_prose'] += events['linked']>0
        metrics['streams_with_complete_claim_prose_question'] += events['complete']>0
        metrics['complete_claim_prose_question_events'] += events['complete']
        for before,after in events['complete_distractor_clauses']:
            metrics['complete_with_intervening_claim_to_prose']+=before>0
            metrics['complete_with_intervening_prose_to_question']+=after>0
            metrics['complete_with_intervening_both']+=(before>0 and after>0)
            metrics['claim_to_prose_distractor_clauses_total']+=before
            metrics['prose_to_question_distractor_clauses_total']+=after
        metrics['streams_with_bracket'] += 'bracket_close' in labs
        metrics['streams_with_complete_triple_and_bracket'] += (
            events['complete']>0 and 'bracket_close' in labs)
        metrics['narrative_source_object_sites'] += sum(
            bool(gen.source_object_candidates(words[:t])) for t in range(length))
        for t,(word,q) in enumerate(zip(words,qs)):
            assert abs(sum(q.values())-1)<1e-9 and q.get(word,0)>0
            assert q==gen.next_distribution(words[:t])
            if labs[t]=='mixed_object':
                parts=gen.object_rule_components(words[:t])
                rebuilt=Counter()
                for part in parts.values():
                    for token,p in part['distribution'].items():
                        rebuilt[token]+=part['mass']*p
                assert set(q)==set(rebuilt)
                assert max(abs(q[k]-v) for k,v in rebuilt.items())<1e-10
                metrics['audited_object_component_sites']+=1
                metrics['source_component_mass_sum']+=parts.get(
                    'source_fact',{}).get('mass',0.)
                metrics['linked_component_mass_sum']+=parts.get(
                    'linked_claim',{}).get('mass',0.)
            history, raw=_split(words[:t])
            content=_content(raw)
            state=linked_state(history)
            if content and content[0] not in ('As','Who','Section','Detail',
                                               'What','Where','How'):
                frame=_frame(content)
                if (state['pending'] and len(content)==OBJECT_INDEX[frame]
                    and content[_indices(frame)[0]]==state['pending'][1]):
                    metrics['pending_claim_matching_subject_object_sites']+=1
                    assert q[state['pending'][0]]>gen.linked_object_mass
            if content and content[0]=='Who' and len(content)==3 and state['queryable']:
                obj=state['queryable'][0]
                assert q[obj]>=gen.linked_query_key_mass
                metrics['linked_source_query_key_sites']+=1
        if len(samples)<4 and events['complete']:
            samples.append(dict(stream=stream,complete=events['complete_objects'],
                                text=' '.join(words)))
    return dict(config=dict(n=n,length=length,seed=seed,kwargs=kwargs),
                labels=dict(labels),metrics=dict(metrics),samples=samples)


def baseline_panel(n,length,seed,kwargs):
    excluded=('linked_subject_mass','linked_object_mass',
              'linked_question_mass','linked_query_key_mass')
    gen=SurfaceMixtureV9SourceBridge(seed=seed,**{k:v for k,v in kwargs.items()
                                                  if k not in excluded})
    metrics=Counter()
    for _ in range(n):
        ids,labs,_=gen.sample_sequence(length)
        words=[gen.id2token[t] for t in ids]
        events=completed_event_triples(words)
        metrics['streams_with_claim']+=events['claims']>0
        metrics['streams_with_linked_prose']+=events['linked']>0
        metrics['streams_with_complete_claim_prose_question']+=events['complete']>0
        metrics['complete_claim_prose_question_events']+=events['complete']
        for before,after in events['complete_distractor_clauses']:
            metrics['complete_with_intervening_claim_to_prose']+=before>0
            metrics['complete_with_intervening_prose_to_question']+=after>0
            metrics['complete_with_intervening_both']+=(before>0 and after>0)
        metrics['streams_with_bracket']+='bracket_close' in labs
        metrics['streams_with_complete_triple_and_bracket']+=(
            events['complete']>0 and 'bracket_close' in labs)
        metrics['narrative_source_object_sites']+=sum(
            bool(gen.source_object_candidates(words[:t])) for t in range(length))
    return dict(seed=seed,metrics=dict(metrics))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=100)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=98001)
    p.add_argument('--regime',choices=('diagnostic','reference_candidate'),
                   default='reference_candidate')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.regime=='diagnostic':
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
    kwargs.update(linked_subject_mass=.65,linked_object_mass=.65,
                  linked_question_mass=.30,linked_query_key_mass=.65)
    rows=[panel(a.n,a.length,a.seed+j,kwargs) for j in (0,1)]
    baseline=[baseline_panel(a.n,a.length,a.seed+j,kwargs) for j in (0,1)]
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(config=vars(a),panels=rows,
      baseline_v9_same_seeds=baseline,
      caveat='Rates are provisional, not FineWeb-calibrated. Complete triple '
             'requires claim -> matching completed prose -> later question '
             'in that temporal order.'),indent=2)+'\n')
    print([r['metrics'] for r in rows])
    print('v9 baseline', [r['metrics'] for r in baseline])


if __name__=='__main__':main()
