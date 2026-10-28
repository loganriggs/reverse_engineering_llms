"""Fixed two-preamble competence screen before context-transfer interventions."""
import argparse, hashlib, json
from pathlib import Path
import torch
from inspect_model import load_model, trace, encode
from probe_training_task_responses import competence
P = Path(__file__).resolve().parent
PREAMBLES = ['Consider the following text. ', 'Read this carefully: ']
FAMILIES = ['agreement', 'sentiment', 'plural_noun', 'past_tense']

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    inp = P/'validation/training_source_context_audit.json'
    panel = json.loads(inp.read_text())
    pairs = [p for p in panel['rows'] if p['family'] in FAMILIES]
    assert len(pairs) == 36
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    model, receipt = load_model()
    device = next(model.parameters()).device
    rows, samples = [], []
    for preamble_index, preamble in enumerate(PREAMBLES):
        for pair in pairs:
            texts = [preamble+t for t in pair['texts']]
            ids = [encode(t, device) for t in texts]
            tokens = [x[0].tolist() for x in ids]
            equal = len(tokens[0]) == len(tokens[1])
            changed = [i for i,(a,b) in enumerate(zip(*tokens)) if a != b]
            endpoints = []
            for endpoint in (0,1):
                logits, _ = trace(model, ids[endpoint])
                lg = logits[0,-1]
                endpoints.append(competence(lg, pair['answer_groups'][endpoint], int(pair['variants'][endpoint])))
                if pair == pairs[0]:
                    samples.append(dict(row=len(rows), endpoint=endpoint, logits=lg.cpu(), groups=pair['answer_groups'][endpoint], variant=int(pair['variants'][endpoint])))
            rows.append(dict(id=pair['id'], family=pair['family'], preamble_index=preamble_index, texts=texts, token_ids=tokens,
                equal_length=equal, changed_positions=changed, single_token_edit=equal and len(changed)==1,
                endpoints=endpoints, both_binary=all(x['binary'] for x in endpoints), top1_rate=sum(x['top1'] for x in endpoints)/2))
        print('completed preamble', preamble_index, flush=True)
    cohorts=[]
    for preamble_index in range(2):
        for family in FAMILIES:
            rr=[r for r in rows if r['family']==family and r['preamble_index']==preamble_index]
            binary=sum(r['both_binary'] for r in rr)/len(rr)
            top1=sum(r['top1_rate'] for r in rr)/len(rr)
            cohorts.append(dict(family=family,preamble_index=preamble_index,n=len(rr),both_endpoint_binary_rate=binary,full_vocab_top1_rate=top1,
                competence_gate_pass=binary>=.75 and top1>=.5,all_single_token=all(r['single_token_edit'] for r in rr)))
    result=dict(status='completed',preambles=PREAMBLES,families=FAMILIES,forward_count=len(rows)*2,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
        checkpoint_sha256=receipt.weights_sha256,cohorts=cohorts,rows=rows,
        protocol='All 36 original training pairs in four P7-selected families, with exactly two fixed preambles. No error filtering or prompt repair. Cohort gates inherited: >=75% both-endpoint binary and >=50% full-vocabulary top1. Exploratory training variants, not held-out tasks.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    torch.save(samples,'/tmp/emlp_p8_competence_logits.pt')
    print(json.dumps(cohorts,indent=2))
if __name__=='__main__': main()
