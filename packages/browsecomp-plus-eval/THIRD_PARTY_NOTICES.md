# Third-party notices

## vLLM-RWKV

`vendor/vllm-rwkv/` is a physical source snapshot of the vLLM-RWKV runtime
used by this evaluation stack. Its Apache License 2.0 text is retained at
`vendor/vllm-rwkv/LICENSE`. Preserve that license and upstream notices when
redistributing the package.

## BrowseComp-Plus assets

The official dataset, decrypted corpus, BM25 index, qrels, model weights, and
official evaluator are not included because they are large and/or separately
distributed. Users must obtain them from `texttron/BrowseComp-Plus` and follow
the upstream terms.

## Evaluation harness

The source checkout used to assemble this package did not contain a top-level
license file for the RWKV evaluation harness. Confirm and add the intended
license before publishing this package as a public GitHub repository.
