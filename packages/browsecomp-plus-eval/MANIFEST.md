# Extraction manifest

- Source checkout: `rwkv-skills`
- Source branch: `upload/g1h-eval-runtime-20260720`
- Source commit: `54639619f859`
- Extraction date: `2026-07-21`
- Extraction mode: physical files, no symlinks to the parent checkout

Included:

- Complete `src/` runtime snapshot so transitive runner, DB, dataset-prepper,
  prompt-format, inference-client, and audit imports remain reproducible.
- Complete `vendor/vllm-rwkv/` source snapshot, including its license.
- BrowseComp-Plus configs, parallel-candidate experiment scripts, official-run
  exporter, audit restoration helper, and focused tests.
- Safe parameterized launch, endpoint probe, asset preflight, and official
  evaluator wrapper scripts.

Excluded:

- Database contents, task/completion/score rows, `.env`, logs, credentials,
  model weights, official corpus/index/qrels, and generated results.
- Frontend, scheduler deployment state, screen sessions, and runner state.

The copied `src/` reflects the working-tree implementation at extraction time,
including local BrowseComp-Plus fixes that may be newer than the named commit.
Machine-specific endpoint paths, task IDs, and default API tokens were removed
from the package copies; callers must pass them explicitly through environment
variables or CLI arguments.

The vendored vLLM files are byte-identical to the source snapshot except for
four additive `.gitignore` negations that ensure source/test files matched by
broad upstream ignore patterns are included by an ordinary `git add`.

Key source checksums at extraction time:

- BrowseComp-Plus runner: `804de0731cb2a1cd0b03e5c084b720e7c14f9c47f5e362ac6bfe3c0aca8e57b0`
- parallel-candidate router: `6df4a847e8aa9088d38fa3870d7e5efe73d18ab855e0ca15caf663364b3b3d50`
- vendored vLLM license: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
