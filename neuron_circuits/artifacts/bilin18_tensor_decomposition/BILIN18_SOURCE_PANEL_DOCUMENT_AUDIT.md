# Document dependence audit

Corpus builder appends token50256 after each document then packs consecutive513-token rows. Relative document IDs inferred from all EOS tokens. Literal EOS strings could introduce extra apparent boundaries. No EOS in a region proves no appended document boundary there.

| Panel | Rows | Documents at measured sites | Largest document share |
|---|---:|---:|---:|
| H235 | 32 | 20 | 0.266 |
| H236 | 64 | 44 | 0.156 |
| H237 | 32 | 16 | 0.219 |
| H238 | 32 | 19 | 0.312 |
| H239 | 32 | 12 | 0.438 |
| H240 | 32 | 1 | 1.000 |
| H241 | 32 | 1 | 1.000 |

H240 and H241 are disjoint token rows within the same source document, not independent document replications. Row-bootstrap intervals describe conditional within-panel resampling, not document-population uncertainty. Next replication must sample distinct documents and preserve document IDs.
