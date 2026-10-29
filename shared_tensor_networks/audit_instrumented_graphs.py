"""Independent explicit-DAG interpreter verifies graph interventions and counts."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from recover_instrumented_graphs import evaluate, resources


def graph(bits):
    p,i,s=bits;nodes={}
    def mul(name,left,right):nodes[name]=(left,right);return name
    hl=mul('h_left','input:u0','input:u1')
    hr=hl if s else mul('h_right','input:u0','input:u1')
    vl=mul('v_left','input:v0','input:v1')
    vr=mul('v_right','input:v0' if i else 'input:w0','input:v1' if i else 'input:w1')
    gl=mul('g_left','parameter:left',vl)
    gr=mul('g_right','parameter:left' if p else 'parameter:right',vr)
    outputs=[mul('y_left',hl,gl),mul('y_right',hr,gr)]
    return nodes,outputs


def execute(bits,x,theta,interventions):
    nodes,outputs=graph(bits)
    values={'input:'+key:x[:,j].copy() for j,key in enumerate(['u0','u1','v0','v1','w0','w1'])}
    values.update({'parameter:left':np.full(len(x),theta[0]),'parameter:right':np.full(len(x),theta[1])})
    if 'parameter' in interventions:values['parameter:left'][:]=1.1
    if 'leaf' in interventions:values['input:v0']+=.5
    if 'state' in interventions:values['h_left']=np.full(len(x),1.3)
    def get(name):
        if name not in values:
            left,right=nodes[name];values[name]=get(left)*get(right)
        return values[name]
    return np.column_stack([get(name) for name in outputs])


def main():
    root=Path(__file__).parent;out=root/'results/instrumented_graph_recovery'
    original=json.loads((out/'results.json').read_text());assert len(original['records'])==1280
    rng=np.random.default_rng(86000);maxerr=0.;checks=0
    for bits in itertools.product([False,True],repeat=3):
        nodes,outputs=graph(bits);leaves={key for pair in nodes.values() for key in pair if ':' in key}
        actual=dict(distinct_parameters=sum(k.startswith('parameter:') for k in leaves),
                    distinct_input_variables=sum(k.startswith('input:') for k in leaves),multiplication_nodes=len(nodes),
                    arithmetic_operand_edges=2*len(nodes))
        assert actual==resources(bits)
        for _ in range(20):
            x=rng.normal(size=(32,6));theta=rng.uniform(.1,1.2,size=2)
            for mask in range(8):
                probes=[name for j,name in enumerate(['parameter','leaf','state']) if mask&(1<<j)]
                expected=evaluate(bits,x,theta,probes);observed=execute(bits,x,theta,probes)
                error=float(np.max(np.abs(expected-observed)));maxerr=max(maxerr,error);assert error<1e-12;checks+=1
    for row in original['records']:
        # Independent combinatorial criterion: each known instrument reveals exactly its bit.
        indices=[['parameter','leaf','state'].index(p) for p in row['probe_subset']]
        possible=[list(b) for b in itertools.product([False,True],repeat=3) if all(b[j]==row['truth'][j] for j in indices)]
        assert possible==row['consistent_candidates']
        best=min(possible,key=lambda b:sum(resources(b).values()));assert best==row['complexity_selected']
    result=dict(status='passed',explicit_DAG_checks=checks,max_direct_interpreter_error=maxerr,identification_sets_checked=1280,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
