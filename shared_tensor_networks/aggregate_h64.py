"""Predeclared teacher-level aggregation; no file IO or model selection."""
import numpy as np

def aggregate(rows):
    seeds=list(range(256101,256106));inits=['257001','257002','teacher'];ns=[8192,32768];tasks=[520,2080]
    expected={(s,n,t,i) for s in seeds for n in ns for t in tasks for i in inits}
    index={(r['teacher'],r['n'],r['tasks'],r['initialization']):r for r in rows}
    assert len(rows)==len(index)==60 and set(index)==expected
    for r in rows:
        assert len(r['component_cosines'])==64 and len(r['population_mse'])==r['tasks']
        assert len(r['transfer']['learned_8192']['errors'])==4
        assert np.isfinite(r['component_cosines']).all() and np.isfinite(r['population_mse']).all()
        assert np.isfinite(r['transfer']['learned_8192']['errors']).all() and np.isfinite(r['train_mse'])
    conditions=[];witnesses=[]
    for n in ns:
      for t in tasks:
       for kind,initializations in [('random',inits[:2]),('teacher',['teacher'])]:
        selected=[index[s,n,t,i] for s in seeds for i in initializations]
        teachers=[]
        for seed in seeds:
            rr=[index[seed,n,t,i] for i in initializations]
            teachers.append(dict(teacher=seed,transfer=float(np.mean([np.mean(r['transfer']['learned_8192']['errors']) for r in rr])),
                                 source=float(np.mean([np.mean(r['population_mse']) for r in rr]))))
        values=[v['transfer'] for v in teachers]
        conditions.append(dict(n=n,tasks=t,labels=n*t,initialization=kind,teachers=teachers,median_transfer=float(np.median(values)),
          min_teacher_transfer=min(values),max_teacher_transfer=max(values),median_source=float(np.median([v['source'] for v in teachers])),
          recovery=sum(min(r['component_cosines'])>.99 for r in selected),runs=len(selected)))
    def by_teacher(n,t):return np.array([g['transfer'] for c in conditions if c['n']==n and c['tasks']==t and c['initialization']=='random' for g in c['teachers']])
    equal_diff=by_teacher(32768,520)-by_teacher(8192,2080)
    fixed={str(n):dict(more_task_wins=int(np.sum(by_teacher(n,2080)<by_teacher(n,520))),teacher_differences=(by_teacher(n,2080)-by_teacher(n,520)).tolist()) for n in ns}
    for seed,n,t,init in sorted(expected):
        if init=='teacher':continue
        r=index[seed,n,t,init];teacher=index[seed,n,t,'teacher'];gap=r['train_mse']-teacher['train_mse']
        witnesses.append(dict(candidate=r['name'],lower_loss_candidate=teacher['name'],empirical_gap=float(gap),lower_by_margin=gap>1e-8))
    return dict(conditions=conditions,equal_budget=dict(labels=17039360,more_inputs_wins=int(np.sum(equal_diff<0)),teacher_differences=equal_diff.tolist()),
                fixed_input=fixed,lower_loss_witnesses=witnesses)
