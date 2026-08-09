# G1i strict-46 root provisioning contract

Status: **production fail-closed; this bundle does not deploy or start
anything.** `provision_root_runtime.sh` never connects to 157/8222, changes a
database, enables a unit, launches an inference server, or touches a GPU.

## Fixed trust boundary

| Object | Required location/identity |
| --- | --- |
| scheduler user/group | `rwkv-strict46:rwkv-strict46` (independent system UID) |
| frozen source runtime | `/opt/rwkv-strict46/runtime/<manifest_sha256>` |
| frozen Python runtime | `/opt/rwkv-strict46/python/<reviewed_runtime_id>` |
| per-run mutable state | `/var/lib/rwkv-strict46/<run-id>` |
| optional service log root | `/var/log/rwkv-strict46` |
| public host-role marker | `/etc/rwkv-strict46/host-role` |

The frozen source and Python trees are root-owned, symlink-free, group
read/execute only, and content-addressed. The service UID can write only its
private per-run state. `run_model.sh`, audit recovery, PID files, and logs do
not read or write any development checkout.

The frozen manifest is schema `rwkv.g1i-strict46-frozen-runtime.v2`. It binds
this guide, the provisioning script, public templates, the exact runtime
scripts, all strict-46 sources/configs/datasets, the Python tree, approval, and
lock. Legacy chain scripts are not publication inputs.

## Publication input policy

Publication is an exact allowlist copy. Any source or Python-runtime symlink,
top-level `.env` (case-insensitive), `*.key`, `.pgpass`, `.netrc`, private-key
name, or credential/token/secret-like data filename makes publication fail.
Only dataset manifests may appear in `support_files`; they must remain under
the frozen `data/` tree. The manifest contains hashes and public metadata, not
secret values.

Use a reviewed, root-owned release copy. Do not copy credentials, database
DSNs, Judge keys, SSH private keys, or shell environment files into it.

## Handoff and systemd contract

Unprivileged waiters no longer call a per-user service manager or create
transient user units.
They perform the frozen gate/audit/idle checks and write a content-addressed
request under:

```text
/var/lib/rwkv-strict46/<run-id>/handoff-requests/<request_sha256>.json
```

They return exit status 75 and do **not** stop/start a model or scheduler. A
separate root-owned system orchestrator must independently revalidate the
request, the frozen manifest, current DB audit, endpoint attestation, GPU/port,
and the fixed transition allowlist before using pre-installed system units.
Giving the service UID arbitrary `sudo systemctl` access is forbidden.

The administrator must fill these **non-secret** deployment values in the
review record before installing any system unit:

```text
HOST_ROLE=__157_OR_8222__
FROZEN_MANIFEST_SHA256=__64_LOWERCASE_HEX__
PYTHON_RUNTIME_ID=__REVIEWED_ID__
PYTHON_INVENTORY_SHA256=__64_LOWERCASE_HEX__
GLOBAL_APPROVAL_SHA256=__64_LOWERCASE_HEX__
PROTOCOL_LOCK_SHA256=__64_LOWERCASE_HEX__
RUN_ID=__SAFE_RUN_ID__
SYSTEM_INFERENCE_UNIT=__ROOT_OWNED_FIXED_UNIT__
SYSTEM_SCHEDULER_UNIT=__ROOT_OWNED_FIXED_UNIT__
EXPECTED_MODEL=__APPROVED_G1I_MODEL__
EXPECTED_GPU=__APPROVED_GPU_INDEX__
EXPECTED_PORT=__APPROVED_LOOPBACK_PORT__
```

Passwords, tokens, Judge API keys, private-key bytes, and database connection
secrets are deliberately absent.

## Read-only review

```bash
sha256sum \
  ops/g1i_strict46/provision_root_runtime.sh \
  ops/g1i_strict46/ROOT_PROVISIONING.md \
  ops/g1i_strict46/templates/ssh_config.157-to-8222.in \
  ops/g1i_strict46/templates/known_hosts.157-to-8222.in \
  ops/g1i_strict46/templates/strict46_db_grants.sql.in
bash -n ops/g1i_strict46/provision_root_runtime.sh
python -m pytest -q tests/test_g1i_root_provisioning_bundle.py
ops/g1i_strict46/provision_root_runtime.sh plan --host-role 157
ops/g1i_strict46/provision_root_runtime.sh plan --host-role 8222
```

`plan` is read-only. Every mutating provisioner phase requires root, `--apply`,
and the literal acknowledgement
`I_UNDERSTAND_STRICT46_ROOT_PROVISIONING`.

## Prepare the independent UID and roots

Run locally on each reviewed host:

```bash
sudo ops/g1i_strict46/provision_root_runtime.sh prepare-host \
  --host-role __157_OR_8222__ --apply \
  --ack I_UNDERSTAND_STRICT46_ROOT_PROVISIONING
```

Existing objects with unexpected owner, group, mode, role, or type cause a
failure; the script does not silently repair unknown state.

## Seal Python and publish the frozen runtime

Prepare an offline `venv --copies` with reviewed wheels at
`/opt/rwkv-strict46/python/<reviewed_runtime_id>`. It must contain no symlink.
Inventory and seal it only after two-host digest review:

```bash
sudo ops/g1i_strict46/provision_root_runtime.sh inventory-python \
  --python-runtime-id __REVIEWED_ID__

sudo ops/g1i_strict46/provision_root_runtime.sh seal-python \
  --host-role __157_OR_8222__ \
  --python-runtime-id __REVIEWED_ID__ \
  --python-executable-relative bin/python3 \
  --expected-python-inventory-sha256 __64_LOWERCASE_HEX__ \
  --apply --ack I_UNDERSTAND_STRICT46_ROOT_PROVISIONING
```

After an independent candidate build supplies the expected manifest SHA:

```bash
sudo ops/g1i_strict46/provision_root_runtime.sh publish-runtime \
  --host-role __157_OR_8222__ \
  --repo /root/release/rwkv-skills-strict46 \
  --approval /root/release/rwkv-skills-strict46/ops/g1i_strict46/approvals/__APPROVAL__.json \
  --approval-sha256 __64_LOWERCASE_HEX__ \
  --lock /root/release/rwkv-skills-strict46/ops/g1i_strict46/protocol_gate.lock.json \
  --lock-sha256 __64_LOWERCASE_HEX__ \
  --python-runtime-id __REVIEWED_ID__ \
  --python-executable-relative bin/python3 \
  --expected-python-inventory-sha256 __64_LOWERCASE_HEX__ \
  --expected-manifest-sha256 __64_LOWERCASE_HEX__ \
  --apply --ack I_UNDERSTAND_STRICT46_ROOT_PROVISIONING
```

Publication is no-clobber and atomic. A pre-existing target is accepted only
after the full frozen gate, ownership/mode, manifest digest, Python contract,
and secret-path checks pass.

## SSH and database prerequisites

Public SSH config/known-host templates and the least-privilege DB grant
template are included in the manifest. Secret material is installed only by an
administrator through an independent secure channel. Prefer local PostgreSQL
peer authentication for the `rwkv-strict46` role; never place a password or
DSN in the frozen runtime or handoff request.

No system service may be enabled until the administrator has reviewed the
filled non-secret values, installed fixed root-owned units, and implemented a
request consumer that fails closed on every mismatch.
