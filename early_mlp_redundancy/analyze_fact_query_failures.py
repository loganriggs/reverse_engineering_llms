"""Post-screen behavior description, without repairing prompts or selecting examples."""
import json
from collections import defaultdict
from fit_pca_encoder import V,sha


def main():
    pp=V/'fact_query_panel_protocol.json';rp=V/'fact_query_panel_screen.json';a=json.loads((V/'fact_query_panel_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);p=json.loads(pp.read_text());r=json.loads(rp.read_text());meta={x['id']:x for x in p['rows']};observed={x['id']:x for x in r['rows']};summaries=[]
    for cohort in r['summaries']:
        ids={x['id'] for x in p['rows'] if x['domain']==cohort['domain'] and x['format']==cohort['format']};counts={k:0 for k in ['queried_value','other_entity_value','unmentioned_value','first_mentioned_value','last_mentioned_value']}
        for key in ids:
            m=meta[key];pred=observed[key]['predicted_index'];av=m['a'];bv=2+m['b'];counts['queried_value']+=pred==m['correct_index'];counts['other_entity_value']+=pred==(bv if m['query']==0 else av);counts['unmentioned_value']+=pred not in [av,bv];counts['first_mentioned_value']+=pred==(av if m['order']==0 else bv);counts['last_mentioned_value']+=pred==(bv if m['order']==0 else av)
        assert counts['queried_value']+counts['other_entity_value']+counts['unmentioned_value']==32
        edge_stats=[]
        for factor in ['a','b','query','order']:
            edges=[e for e in p['edges'] if e['recipient_id'] in ids and e['factor']==factor];assert len(edges)==16
            for relevant in [True,False]:
                es=[e for e in edges if e['answer_should_change']==relevant]
                if not es:continue
                edge_stats.append(dict(factor=factor,intended_answer_changes=relevant,n=len(es),candidate_choice_changes=sum(observed[e['recipient_id']]['predicted_index']!=observed[e['donor_id']]['predicted_index'] for e in es),both_endpoints_correct=sum(observed[e['recipient_id']]['four_way_correct'] and observed[e['donor_id']]['four_way_correct'] for e in es)))
        # Independent regrouping of rows verifies query-switch pair accounting.
        grouped=defaultdict(list)
        for key in ids:
            m=meta[key];grouped[m['case'],m['a'],m['b'],m['order']].append(key)
        assert len(grouped)==16 and all(len(v)==2 for v in grouped.values());direct=sum(observed[a]['predicted_index']!=observed[b]['predicted_index'] for a,b in grouped.values());query=next(x for x in edge_stats if x['factor']=='query');assert direct==query['candidate_choice_changes']
        summaries.append(dict(domain=cohort['domain'],format=cohort['format'],n=32,counts=counts,edges=edge_stats))
    out=dict(status='completed',source_sha256=sha(__file__),protocol_sha256=sha(pp),screen_sha256=sha(rp),audit_sha256=sha(V/'fact_query_panel_audit.json'),summaries=summaries,scope='Explicit post-result descriptive accounting across all6failedcohorts. Choices among4declared answer tokens only; notnecessarily fullvocabularytop1. Query/order/factedges are promptchanges, notinternal causalpatches. No fitted heuristic orsemantic relabeling; no prompts/gates changed. Candidatechoice stability doesnot implyentire outputdistribution is unchanged.')
    path=V/'fact_query_panel_failures.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
    for row in summaries:print(row['domain'],row['format'],row['counts'],'QUERY',next(x for x in row['edges'] if x['factor']=='query'))
if __name__=='__main__':main()
