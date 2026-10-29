"""M108 fixed conflicting-value factorial panel; empirical behavior, not correctness labels."""
import json
import tiktoken
from fit_pca_encoder import V,sha
from repetition_cube_algebra import BITS


def main():
    tok=tiktoken.get_encoding('gpt2');rows=[];specs=[('white','red','yellow'),('orange','blue','black'),('purple','green','white'),('red','yellow','orange')];old=json.loads((V/'whole_task_native_protocol.json').read_text())
    for pair in old['pairs']:
        if pair['family']=='sequence_copy':
            for ep in [0,1]:rows.append(dict(id=pair['id']+f'|{ep}',cohort='historical_replay',token_ids=pair['token_ids'][ep],answer_groups=pair['answer_groups'][ep],correct_index=pair['variants'][ep]))
    for case,(key,v0,v1) in enumerate(specs):
        for fmt,sep in [('semicolon','; '),('comma',', ')]:
            for vertex,bits in enumerate(BITS.tolist()):
                values=[v0,v1];prompt=sep.join([key+' '+values[b] for b in bits]+[key]);ids=tok.encode(prompt);answers=[];groups=[]
                for value in values:
                    ext=tok.encode(prompt+' '+value);assert ext[:-1]==ids and len(ext)==len(ids)+1;answers.append(ext[-1]);g=[]
                    for form in set([value,value.capitalize(),value.upper()]):
                        ext=tok.encode(prompt+' '+form)
                        if ext[:-1]==ids and len(ext)==len(ids)+1:g.append(ext[-1])
                    groups.append(sorted(set(g)))
                rows.append(dict(id=f'm108:{case}:{fmt}:{vertex}',cohort='cube',case=case,format=fmt,key=key,values=values,bits=bits,vertex=vertex,prompt=prompt,token_ids=ids,canonical_answer_tokens=answers,answer_groups=groups))
    assert len(rows)==80 and len({tuple(x['token_ids']) for x in rows})==80
    p=dict(status='frozen_before_native_outputs',source_sha256=sha(__file__),toy_sha256=sha(V/'repetition_cube_algebra_toys.json'),historical_protocol_sha256=sha(V/'whole_task_native_protocol.json'),rows=rows,primary='Canonical value1 minusvalue0 final logit margin. Uniform3bit Walsh decomposition, all8corners, all8cubes. Position coefficients andhigherorder energies measured; no correctanswer assigned toconflicting sequences.',secondary='Casevariant-pooled candidate decisions andvocabtop1 membership separately. First/last/majority heuristic agreement withnativechoices, includingconsistent andmixedvertices separately. A thresholded linear margin canyieldmajority choices withoutmargin interaction.',native_mlp='Final-position normalized inputs andproductwrites atMLP0/1/2: vectorWalshcoefficients, exact nativebilinear XORconvolution; cubic term splitmean×cubicinput versuslinear×pairinput, signedcrossenergy retained. This is functioncomposition acrosspromptfactors, notnativechannelcancellation orcausalnecessity.',featurizers='Later fixed-model comparison may scoretheseempirical nativefactor-response targets. Entiresequencecopyfamily remains excludedfromfeaturizer fitting; no labels/modelparameters/selection trained oncube.',stopping='One fixedpanel andalgebraicnativeanalysis, no sequence/punctuation/lexicalsearch afteroutcomes. No semanticretrieval claim onambiguoussequences. No nativecomponent intervention thisstage; anylatercausal/featurizertest separatelyfrozen.')
    f=V/'repetition_cube_protocol.json';assert not f.exists();f.write_text(json.dumps(p,indent=2)+'\n');print('FROZEN 64cube +16replay')
if __name__=='__main__':main()
