# External assets

The model weights, the 2+ GB decrypted BrowseComp-Plus JSONL, the BM25 index,
qrels, and upstream evaluator are intentionally not duplicated here. Obtain
them from the official `texttron/BrowseComp-Plus` distribution under its
terms, then set `BROWSECOMP_PLUS_ROOT` to a directory containing:

```text
BrowseComp-Plus/
├── data/browsecomp_plus_decrypted.jsonl
├── indexes/bm25/
├── scripts_evaluation/evaluate_run.py
└── topics-qrels/qrel_evidence.txt
```

Prepared manifests written below `data/browsecomp_plus/` are ignored by Git.
