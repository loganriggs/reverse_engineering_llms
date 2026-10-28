"""H1041 adaptive reserved-task format repair, fixed before this screen."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
from transformers import AutoTokenizer
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_reserved_task_format_screen';out=A/'reserved_task_format_screen';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
import run_subject_number_sparse_graph_token_extraction_v1 as runner
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
tok=AutoTokenizer.from_pretrained('/home/loganriggs/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e',local_files_only=True)
specs=[]
def add(family,split,template,items):
 assert len(items)==8
 specs.append((family,split,[(template.format(x=x),y) for x,y in items]))
add('weekday_successor__sequence','development','{x},',[('Monday, Tuesday','Wednesday'),('Tuesday, Wednesday','Thursday'),('Wednesday, Thursday','Friday'),('Thursday, Friday','Saturday'),('Friday, Saturday','Sunday'),('Saturday, Sunday','Monday'),('Sunday, Monday','Tuesday'),('monday, tuesday','Wednesday')])
add('weekday_successor__prose','development','Yesterday was {x}. Today is',[('Monday','Tuesday'),('Tuesday','Wednesday'),('Wednesday','Thursday'),('Thursday','Friday'),('Friday','Saturday'),('Saturday','Sunday'),('Sunday morning','Monday'),('Monday morning','Tuesday')])
ordinal=[('two','second'),('three','third'),('four','fourth'),('five','fifth'),('six','sixth'),('seven','seventh'),('eight','eighth'),('nine','ninth')]
add('ordinal_word__analogy','development','one: first; {x}:',ordinal)
add('ordinal_word__list','development','one, first; {x},',ordinal)
animal=[('robin','bird'),('salmon','fish'),('eagle','bird'),('trout','fish'),('duck','bird'),('bass','fish'),('swan','bird'),('carp','fish')]
add('animal_category__type','development','A {x} is a type of',animal)
add('animal_category__example','development','A {x} is an example of a',animal)
syn=[('quick','fast'),('silent','quiet'),('happy','glad'),('angry','mad'),('clever','smart'),('ill','sick'),('large','big'),('small','little')]
add('synonym__analogy','development','Synonyms: large = big; {x} =',syn)
add('synonym__means','development','Synonyms: large means big; {x} means',syn)
sup=[('good, better','best'),('bad, worse','worst'),('large, larger','largest'),('small, smaller','smallest'),('high, higher','highest'),('low, lower','lowest'),('long, longer','longest'),('short, shorter','shortest')]
add('superlative__sequence','evaluation','{x},',sup)
add('superlative__example','evaluation','Comparisons: good, better, best; {x},',sup)
add('multiplication__words','evaluation','{x} is',[('Two times three','six'),('Three times three','nine'),('Four times two','eight'),('Five times two','ten'),('Three times four','twelve'),('Four times four','sixteen'),('Five times three','fifteen'),('Six times three','eighteen')])
add('multiplication__digits','evaluation','{x} equals',[('2 times 3','6'),('3 times 3','9'),('4 times 2','8'),('5 times 2','10'),('3 times 4','12'),('4 times 4','16'),('5 times 3','15'),('6 times 3','18')])
part=[('go, went','gone'),('eat, ate','eaten'),('write, wrote','written'),('speak, spoke','spoken'),('take, took','taken'),('give, gave','given'),('see, saw','seen'),('do, did','done')]
add('past_participle__sequence','evaluation','{x},',part)
add('past_participle__prose','evaluation','I {x} every day. I have already',[('go','gone'),('eat','eaten'),('write','written'),('speak','spoken'),('take','taken'),('give','given'),('see','seen'),('do','done')])
add('unit_completion__contains','evaluation','{x}',[('There are sixty seconds in a','minute'),('There are sixty minutes in an','hour'),('There are twenty-four hours in a','day'),('There are seven days in a','week'),('There are twelve months in a','year'),('There are one hundred years in a','century'),('A minute contains sixty','seconds'),('An hour contains sixty','minutes')])
add('unit_completion__equation','evaluation','{x}',[('60 seconds = 1','minute'),('60 minutes = 1','hour'),('24 hours = 1','day'),('7 days = 1','week'),('12 months = 1','year'),('100 years = 1','century'),('1 minute = 60','seconds'),('1 hour = 60','minutes')])
assert len(specs)==16
manifest=[];examples={};pairs=[];invalid=[]
for family,split,endpoints in specs:
 for n in range(4):
  pp=endpoints[2*n:2*n+2];row=dict(id=f'{family}-{n}',family=family,base_family=family.split('__')[0],split=split,prompts=[x[0] for x in pp],answers=[x[1] for x in pp]);manifest.append(row);ee=[]
  for variant,(text,answer) in enumerate(pp):
   ids=tok.encode(text,add_special_tokens=False);groups=[]
   for _,a in pp:
    gg=[]
    for form in dict.fromkeys([a,a.capitalize(),a.upper()]):
     comb=tok.encode(text+' '+form,add_special_tokens=False)
     if len(comb)==len(ids)+1 and comb[:len(ids)]==ids:gg.append(comb[-1])
    groups.append(sorted(set(gg)))
   if any(not g for g in groups) or set(groups[0])&set(groups[1]):break
   ee.append(dict(id=row['id'],variant=variant,text=text,ids=ids,groups=groups))
  if len(ee)!=2:invalid.append(row|dict(reason='No disjoint single-token answer groups'));continue
  ii=[len(examples),len(examples)+1];examples.update(zip(ii,ee));pairs.append(row|dict(indices=ii,equal_length=len(ee[0]['ids'])==len(ee[1]['ids']),same_final_token=ee[0]['ids'][-1]==ee[1]['ids'][-1]))
duplicate_prompts=len(examples)-len({e['text'] for e in examples.values()});assert duplicate_prompts==0
config=dict(experiment='H1041',families=8,cohorts=16,declared_pairs=64,split_assignment='Original H1039 four development/four evaluation family assignments preserved. Two fixed format cohorts each; no training eligibility. Adaptive repair informed by H1039, not fresh confirmation.',gate='Per family at least4 token-valid equal-length pairs, >=75% both-endpoint binary accuracy, >=50% intended full-vocabulary top1. Training eligibility determined only by train cohort.',answer_policy='Single-token exact continuation groups for original/capitalized/uppercase answer. Alternatives not listed can make open-vocabulary accuracy conservative.',historical_scope='All H1039 attempts retained. New formats are adaptively constructed; some examples overlap earlier prompts. No novel task discovery or fresh confirmation claim.',evaluation_scope='Native competence screened on all splits; evaluation prompts thus opened for competence, not untouched. No representation fitted or selected.',alignment='Equal token lengths permit positionwise patching but do not prove semantic alignment.',historical_exclusions=['sequence_copy','spatial_inverse'])
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');mh=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
tree=ast.parse((R/'scripts/run_bilin18_projection_stream_factorial.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL','margin']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-functions>','exec'),globals())
states={};native={}
with torch.no_grad():
 for i,e in examples.items():
  initial=F.rms_norm(model.transformer.wte(torch.tensor([e['ids']],device='cuda')),(1152,)).float();raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);st=(raw,x0,bus);native[i]=suffix(st).cpu();states[i]=tuple(v.cpu() for v in st)
records=[]
for p in pairs:
 i,d=p['indices'];correct=[margin(native[k],examples[k]['groups'])*(1 if v==0 else -1)>0 for v,k in enumerate([i,d])];top=[int(native[k].argmax()) in examples[k]['groups'][v] for v,k in enumerate([i,d])];records.append(p|dict(both_correct=all(correct),full_vocab=float(np.mean(top)),margins=[margin(native[k],examples[k]['groups']) for k in [i,d]],top_tokens=[tok.decode([int(native[k].argmax())]) for k in [i,d]]))
summary=[]
for f,sp,_ in specs:
 rr=[r for r in records if r['family']==f and r['equal_length']];both=float(np.mean([r['both_correct'] for r in rr])) if rr else 0;top=float(np.mean([r['full_vocab'] for r in rr])) if rr else 0;passed=len(rr)>=4 and both>=.75 and top>=.5
 summary.append(dict(family=f,split=sp,valid_aligned_pairs=len(rr),both_endpoint_accuracy=both,top1=top,pass_gate=passed,training_eligible=sp=='train' and passed))
# Independent direct full forward at the same precision for one exemplar per family.
replay=0.
with torch.no_grad():
 for f,_,_ in specs:
  matches=[i for i,e in examples.items() if e['id']==f+'-0']
  if not matches:continue
  i=matches[0];x0=F.rms_norm(model.transformer.wte(torch.tensor([examples[i]['ids']],device='cuda')),(1152,)).float();x=x0;bus=None
  for b in model.transformer.h:x,bus=b(x,bus,x0)
  y=(30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30))[0].double().cpu();replay=max(replay,float((y-native[i]).abs().max()))
audit=dict(manifest_unchanged=mh==hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),manifest_sha256=mh,native_direct_replay_max=replay,prefix_sequences=len(examples),suffix_sequences=len(examples),fit_rows=0,all_split_prompts_unique=True,finite=all(bool(torch.isfinite(v).all()) for v in native.values()))
assert replay<1e-5 and audit['manifest_unchanged'] and audit['finite']
torch.save(dict(states=states,native=native,examples=examples),out/'states.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,invalid=invalid,audits=audit),indent=2)+'\n');print(json.dumps(dict(summary=summary,invalid_pairs=len(invalid),audits=audit),indent=2))
