"""Exact-support v8 section-edit panels with whole association holdouts.

An association is (section ID, fact color). Fit/dev exclude heldout
associations from *every* fact in each original prefix. Test-heldout queries
an excluded association; its other facts are unrestricted. The trained LM
has seen all combinations; this is a featurizer-only holdout.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from eval_surface_mixture_v7_scope import build, fact_entries
from surface_mixture_v2 import _split
from surface_mixture_v7 import COLORS, SECTION_IDS
from surface_mixture_v8 import SurfaceMixtureV8


def is_heldout(section,color):
    return (SECTION_IDS.index(section)+COLORS.index(color))%3==0


def annotate(pair,gen):
    words=[gen.id2token[i] for i in pair['original']]
    history,_=_split(words)
    facts=fact_entries(history)
    key=words[-4]
    assert key in facts
    associations=tuple(sorted((section,row['color'])
                              for section,row in facts.items()))
    held=tuple((section,color) for section,color in associations
               if is_heldout(section,color))
    return dict(pair=pair,queried=[key,facts[key]['color']],
                associations=[list(row) for row in associations],
                heldout_associations=[list(row) for row in held],
                fully_seen=not held,
                queried_heldout=is_heldout(key,facts[key]['color']))


def collect(seed,streams,cap,length,kwargs):
    gen=SurfaceMixtureV8(seed=seed,**kwargs)
    questions,pairs,attempted=build(gen,streams,length,cap)
    return [annotate(pair,gen) for pair in pairs],dict(
        n_questions=len(questions),pair_counts=dict(Counter(
            item['kind'] for item in pairs)),attempted=dict(attempted))


def select(rows,mode,cap_per_kind):
    result=[]
    counts=Counter()
    used=set()
    for row in rows:
        pair=row['pair']
        kind=pair['kind']
        if kind not in cap_per_kind or counts[kind]>=cap_per_kind[kind]:
            continue
        if mode=='fit' and (kind!='section_key' or not row['fully_seen']):
            continue
        if mode=='seen' and not row['fully_seen']:
            continue
        if mode=='heldout' and not row['queried_heldout']:
            continue
        key=(kind,tuple(pair['original']),tuple(pair['edited']))
        if key in used:continue
        used.add(key)
        result.append(row)
        counts[kind]+=1
    return result


def check_disjoint(panels):
    seen={}
    for panel,rows in panels.items():
        for row in rows:
            key=tuple(row['pair']['original'])
            if key in seen and seen[key]!=panel:
                raise AssertionError(f'Original prefix appears in {seen[key]} and {panel}')
            seen[key]=panel


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',default='language_task_suite/results/v8_scope_balanced_continue_seed101')
    p.add_argument('--fit-seed',type=int,default=81001)
    p.add_argument('--dev-seed',type=int,default=81002)
    p.add_argument('--test-seed',type=int,default=81003)
    p.add_argument('--fit-streams',type=int,default=1200)
    p.add_argument('--dev-streams',type=int,default=600)
    p.add_argument('--test-streams',type=int,default=1200)
    p.add_argument('--fit-cap',type=int,default=1000)
    p.add_argument('--dev-cap',type=int,default=500)
    p.add_argument('--test-cap',type=int,default=1000)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--nfit',type=int,default=240)
    p.add_argument('--ndev',type=int,default=80)
    p.add_argument('--ntest-key',type=int,default=120)
    p.add_argument('--ntest-color',type=int,default=80)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    saved=json.loads((Path(a.model_dir)/'results.json').read_text())
    assert saved['config']['generator']=='v8'
    kwargs=saved['generator_kwargs']
    raw={}
    counts={}
    for name,seed,n,cap in (('fit',a.fit_seed,a.fit_streams,a.fit_cap),
                            ('dev',a.dev_seed,a.dev_streams,a.dev_cap),
                            ('test',a.test_seed,a.test_streams,a.test_cap)):
        raw[name],counts[name]=collect(seed,n,cap,a.length,kwargs)
        print(name,counts[name],flush=True)
    test_budget=dict(section_key=a.ntest_key,
                     relevant_color=a.ntest_color,
                     irrelevant_color=a.ntest_color)
    panels=dict(fit=select(raw['fit'],'fit',dict(section_key=a.nfit)),
                dev=select(raw['dev'],'fit',dict(section_key=a.ndev)),
                test_seen=select(raw['test'],'seen',test_budget),
                test_heldout=select(raw['test'],'heldout',test_budget))
    check_disjoint(panels)
    assert len(panels['fit'])==a.nfit and len(panels['dev'])==a.ndev
    assert all(len([row for row in panels[name] if row['pair']['kind']==kind])>0
               for name in ('test_seen','test_heldout')
               for kind in test_budget)
    assert all(row['fully_seen'] for name in ('fit','dev','test_seen')
               for row in panels[name])
    assert all(row['queried_heldout'] for row in panels['test_heldout'])
    seen_combos={(section,color) for name in ('fit','dev') for row in panels[name]
                 for section,color in row['associations']}
    held_combos={(section,color) for row in panels['test_heldout']
                 for section,color in [row['queried']]}
    assert not seen_combos & held_combos
    held_universe={(section,color) for section in SECTION_IDS
                   for color in COLORS if is_heldout(section,color)}
    assert seen_combos=={(section,color) for section in SECTION_IDS
                         for color in COLORS if not is_heldout(section,color)}
    assert held_combos==held_universe
    summary={name:dict(n=len(rows),kinds=dict(Counter(
        row['pair']['kind'] for row in rows)),
        queried_combinations=sorted({tuple(row['queried']) for row in rows}),
        all_fact_combinations=sorted({tuple(combo) for row in rows
                                      for combo in row['associations']}))
        for name,rows in panels.items()}
    output=dict(config=vars(a),heldout_universe=sorted(held_universe),
                seen_universe=sorted(seen_combos),raw_counts=counts,
                summary=summary,panels=panels,
                caveat='Whole association split applies to the featurizer only. '
                       'Both frozen language models were trained on all section '
                       'combinations. Test-heldout selects by queried original '
                       'association; color edits may change the queried color.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n')
    print('panel summary',summary,flush=True)


if __name__=='__main__':main()
