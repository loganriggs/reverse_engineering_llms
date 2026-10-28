"""M15 fixed context challenge and native competence screen, no group selection."""
import json,hashlib
from pathlib import Path
import torch,tiktoken
from inspect_model import load_model,trace
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');PREFIXES={'exercise':'A short exercise follows.\n','example':'Here is another example.\n'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');sp=P/'validation/diverse_supervision_split.json';split=json.loads(sp.read_text());bank=json.loads((P/'validation/training_source_context_audit.json').read_text());pairs={x['id']:x for x in bank['rows']};enc=tiktoken.get_encoding('gpt2');existing={text for x in bank['rows'] for text in x['texts']};rows=[]
 for context,prefix in PREFIXES.items():
  for pid in split['evaluation_pair_ids']:
   pair=pairs[pid];texts=[prefix+t for t in pair['texts']];assert not set(texts)&existing;rows.append(dict(context=context,id=pid,family=pair['family'],texts=texts,token_ids=[enc.encode(t) for t in texts],answer_groups=pair['answer_groups'],variants=pair['variants']))
 m13p=P/'validation/group_role_matching.json';m11p=P/'validation/diverse_supervision_results.json';m11=json.loads(m11p.read_text());entries=[x for x in m11['models'] if x['arm']=='mixed_natural_task_KL']
 protocol=dict(source_sha256=sha(Path(__file__)),split_sha256=sha(sp),matching_sha256=sha(m13p),models=entries,prefixes=PREFIXES,rows=rows,competence_gate='Perfamily×context: >=75%pairs havebothendpointsnativebinarycorrect AND >=50%endpointsnativefull-vocabularytop1correct. Gateusesnativeonly; allpairsinpassingcohortretainedinclerrors. No replacements/resamplingifcohortfails. Atleast3passingfamiliesrequiredforcontextaggregate roleprediction;otherwise reportonlycohortoutcomes.',future_causal_test='FrozenM11mixedmodels,all32groups;scalarcontiguous/blockcontiguous/blockshuffled asM13. Removegroupdecodedcontribution withothercodes/gatesfixed,compareintactreconstruction. ScorefrozenM13geometry/function/randomassignments andtrainingamplitudes, no refit. Percontextpassingfamilyset determinesevaluationdimensions; compareoldM13scoresrestrictedtoSAMEfamilies. Reportraw/normalizederrors,energy/cost,intactnativefidelity,andfailedcohorts. No furtherpredictor/model/groupsearch.',scope='Preamblecontexttransfer on14previouslyexaminedlexicalevalpairs,notnewlexicalitems/tasks or independently sampledcorpus. Exactnewstringsabsentfromfocusedtaskbank; scripts/packetsearchfoundnoprefixmention beforecreation,butno claimofglobalhistoricalnovelty. Originalexcludedfamiliesuntouched.')
 pp=P/'validation/context_challenge_protocol.json';pp.write_text(json.dumps(protocol,indent=2)+'\n');model,receipt=load_model();device=next(model.parameters()).device;results=[];samples=[]
 for pair in rows:
  checks=[]
  for ep in [0,1]:
   ids=torch.tensor([pair['token_ids'][ep]],device=device);lg,_=trace(model,ids);check=competence(lg[0,-1],pair['answer_groups'][ep],pair['variants'][ep]);checks.append(check);samples.append(dict(context=pair['context'],id=pair['id'],endpoint=ep,logits=lg[0,-1].cpu(),groups=pair['answer_groups'][ep],variant=pair['variants'][ep]))
  results.append(dict(context=pair['context'],id=pair['id'],family=pair['family'],endpoints=checks,both_binary=all(x['binary'] for x in checks),top1_rate=sum(x['top1'] for x in checks)/2))
 cohorts=[]
 for context in PREFIXES:
  for family in sorted({r['family'] for r in rows}):
   rs=[r for r in results if r['context']==context and r['family']==family];binary=sum(r['both_binary'] for r in rs)/len(rs);top1=sum(r['top1_rate'] for r in rs)/len(rs);cohorts.append(dict(context=context,family=family,pairs=len(rs),both_binary_rate=binary,top1_rate=top1,pass_gate=binary>=.75 and top1>=.5))
 ap=A/'emlp_m15_competence_samples.pt';torch.save(samples,ap);out=dict(status='completed_native_screen',protocol_sha256=sha(pp),checkpoint_sha256=receipt.weights_sha256,forward_count=56,results=results,cohorts=cohorts,context_aggregate_eligible={context:sum(x['pass_gate'] for x in cohorts if x['context']==context)>=3 for context in PREFIXES},samples_path=str(ap),samples_sha256=sha(ap));(P/'validation/context_competence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(cohorts,indent=2))
if __name__=='__main__':main()
