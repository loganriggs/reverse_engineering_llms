#!/usr/bin/env python3
"""Audit document dependence of continuation panels from appended GPT-2 EOS."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
def main():
    corpus=np.load(ROOT/'artifacts/bilin18_layers_0_2/corpus/fineweb_viewer_20k.npy',mmap_mode='r');flat=corpus.reshape(-1);ends=np.flatnonzero(flat==50256)
    panels={'H235':(19720,32),'H236':(19752,64),'H237':(19816,32),'H238':(19848,32),'H239':(19880,32),'H240':(19912,32),'H241':(19944,32)};result={}
    for name,(start,count) in panels.items():
        offsets=np.arange(start,start+count)[:,None]*513+np.array([31,63,95,126])[None]
        docs=np.searchsorted(ends,offsets,side='left');ids,counts=np.unique(docs,return_counts=True);dominant=int(ids[counts.argmax()]);left=int(ends[dominant-1]+1) if dominant else 0;right=int(ends[dominant]) if dominant<len(ends) else len(flat)
        result[name]={'start':start,'rows':count,'site_document_ids':docs.tolist(),'unique_site_documents':len(ids),'dominant_site_fraction':float(counts.max()/counts.sum()),'dominant_relative_document_id':dominant,'dominant_document_start_row_and_offset':list(divmod(left,513)),'dominant_document_end_row_and_offset':list(divmod(right,513)),'eos_in_whole_panel':int((corpus[start:start+count]==50256).sum()),'caution':'Counts identify documents at measurement positions. Attention context can also contain earlier documents within its row.'}
    output={'method':'Corpus builder appends token50256 after each document then packs consecutive513-token rows. Relative document IDs inferred from all EOS tokens. Literal EOS strings could introduce extra apparent boundaries. No EOS in a region proves no appended document boundary there.','panels':result,'conclusion':'H240 and H241 are disjoint token rows within the same source document, not independent document replications. Row-bootstrap intervals describe conditional within-panel resampling, not document-population uncertainty. Next replication must sample distinct documents and preserve document IDs.'}
    (ART/'bilin18_source_panel_document_audit.json').write_text(json.dumps(output,indent=2))
    lines=['# Document dependence audit','',output['method'],'','| Panel | Rows | Documents at measured sites | Largest document share |','|---|---:|---:|---:|']
    for n,v in result.items():lines.append(f"| {n} | {v['rows']} | {v['unique_site_documents']} | {v['dominant_site_fraction']:.3f} |")
    lines+=['',output['conclusion']];(ART/'BILIN18_SOURCE_PANEL_DOCUMENT_AUDIT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
