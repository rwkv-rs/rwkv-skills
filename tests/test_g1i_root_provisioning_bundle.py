from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from ops.g1i_strict46 import require_global_protocol_gate as gate


REPO = Path(__file__).resolve().parents[1]
OPS = REPO / "ops" / "g1i_strict46"
PROVISIONER = OPS / "provision_root_runtime.sh"
GUIDE = OPS / "ROOT_PROVISIONING.md"
SSH_CONFIG_TEMPLATE = OPS / "templates" / "ssh_config.157-to-8222.in"
KNOWN_HOSTS_TEMPLATE = OPS / "templates" / "known_hosts.157-to-8222.in"
DB_GRANTS_TEMPLATE = OPS / "templates" / "strict46_db_grants.sql.in"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PROVISIONER), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_provisioner_is_valid_bash_and_default_is_fail_closed() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(PROVISIONER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    no_args = _run()
    assert no_args.returncode == 64
    assert "Usage:" in no_args.stderr

    unknown = _run("definitely-not-a-phase")
    assert unknown.returncode == 64
    assert "unknown phase" in unknown.stderr


def test_plan_is_read_only_and_role_specific() -> None:
    host_157 = _run("plan", "--host-role", "157")
    assert host_157.returncode == 0, host_157.stderr
    assert "mutations=none" in host_157.stdout
    assert "services_started_or_enabled=never" in host_157.stdout
    assert "ssh_route=157 -> 8222" in host_157.stdout

    host_8222 = _run("plan", "--host-role", "8222")
    assert host_8222.returncode == 0, host_8222.stderr
    assert "mutations=none" in host_8222.stdout
    assert "ssh_route=" not in host_8222.stdout

    invalid = _run("plan", "--host-role", "other")
    assert invalid.returncode == 64


def test_every_mutating_phase_requires_apply_and_exact_ack() -> None:
    script = PROVISIONER.read_text(encoding="utf-8")
    for function in (
        "prepare_host",
        "seal_python",
        "publish_runtime",
        "install_ssh_metadata",
    ):
        body = script.split(f"{function}() {{", 1)[1].split("\n}", 1)[0]
        assert "require_apply_ack" in body

    assert 'readonly APPLY_ACK="I_UNDERSTAND_STRICT46_ROOT_PROVISIONING"' in script
    assert "systemctl" not in script
    assert "systemd-run" not in script
    assert "nvidia-smi" not in script
    assert "psql" not in script
    assert "curl" not in script


def test_runtime_publication_is_content_addressed_no_clobber_and_secret_free() -> None:
    script = PROVISIONER.read_text(encoding="utf-8")
    assert 'readonly RUNTIME_PARENT="$OPT_ROOT/runtime"' in script
    assert 'target="$RUNTIME_PARENT/$expected_manifest"' in script
    assert "expected-manifest-sha256" in script
    assert "mv -T -n" in script
    assert "runtime target appeared during publication" in script
    assert "runtime already published and verified" in script
    assert "scan_runtime_for_secret_paths" in script
    assert 'repo/.env is forbidden in a published runtime' in script
    assert "--frozen-runtime" in script
    assert "--print-frozen-python" in script


def test_release_preflight_rejects_real_env_keys_credentials_and_symlinks(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "release"
    source = repo / "src/module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    published = {"src/module.py"}
    gate.validate_release_source(repo, published_paths=published)

    for relative in (".env", ".ENV", "src/production.key", "src/credentials.json"):
        candidate = repo / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("DO_NOT_PUBLISH_SENTINEL_SECRET", encoding="utf-8")
        with pytest.raises(gate.ProtocolGateError, match="secret-like path"):
            gate.validate_release_source(repo, published_paths=published)
        candidate.unlink()

    target = tmp_path / "outside"
    target.write_text("ordinary", encoding="utf-8")
    root_link = repo / "innocent-link"
    root_link.symlink_to(target)
    with pytest.raises(gate.ProtocolGateError, match="symlink"):
        gate.validate_release_source(repo, published_paths=published)
    root_link.unlink()

    nested_link = repo / "src/nested-link"
    nested_link.symlink_to(target)
    with pytest.raises(gate.ProtocolGateError, match="symlink"):
        gate.validate_release_source(repo, published_paths=published)


def test_python_runtime_requires_root_owned_symlink_free_sealed_tree() -> None:
    script = PROVISIONER.read_text(encoding="utf-8")
    assert 'readonly PYTHON_PARENT="$OPT_ROOT/python"' in script
    assert "symlink refused" in script
    assert "sealed-tree root is a symlink" in script
    assert "cross-filesystem entry in sealed tree" in script
    assert "non-regular Python entry refused" in script
    assert "Python inventory changed while sealing" in script
    assert "find \"$root\" -type f -perm /111 -exec chmod 0550" in script
    assert "find \"$root\" -type f ! -perm /111 -exec chmod 0440" in script
    assert "find \"$root\" -type d -exec chmod 0550" in script
    verify_body = script.split("verify_runtime() {", 1)[1].split("\n}", 1)[0]
    assert 'assert_sealed_tree "$root" "$gid"' in verify_body


def test_publication_uses_only_the_presealed_python_and_rechecks_evidence() -> None:
    script = PROVISIONER.read_text(encoding="utf-8")
    publish_body = script.split("publish_runtime() {", 1)[1].split("\n}", 1)[0]

    assert 'assert_root_owned_source_tree "$repo"' in publish_body
    assert 'assert_sealed_tree "$python_root" "$gid"' in publish_body
    assert 'env PYTHONDONTWRITEBYTECODE=1 "$python_root/$executable_relative" -B -I' in publish_body
    assert publish_body.count('stable_file_sha256 "$approval"') >= 2
    assert publish_body.count('stable_file_sha256 "$lock"') >= 2


def test_ssh_template_pins_trust_inputs_and_remains_unusable_template() -> None:
    config = SSH_CONFIG_TEMPLATE.read_text(encoding="utf-8")
    known_hosts = KNOWN_HOSTS_TEMPLATE.read_text(encoding="utf-8")
    script = PROVISIONER.read_text(encoding="utf-8")

    assert "Host strict46-8222" in config
    assert "HostName __STRICT46_8222_HOST_OR_IP__" in config
    assert "IdentityFile /etc/rwkv-strict46/id_ed25519_8222" in config
    assert "UserKnownHostsFile /etc/rwkv-strict46/known_hosts_8222" in config
    assert "GlobalKnownHostsFile /dev/null" in config
    for setting in (
        "BatchMode yes",
        "IdentitiesOnly yes",
        "StrictHostKeyChecking yes",
        "PasswordAuthentication no",
        "KbdInteractiveAuthentication no",
        "ForwardAgent no",
        "ControlMaster no",
        "PermitLocalCommand no",
    ):
        assert setting in config
    assert "__STRICT46_8222_HOSTKEY_LINE_FROM_TRUSTED_CONSOLE__" in known_hosts
    assert "reject_placeholders" in script
    assert "private key was not created or copied" in script
    assert "expected-identity-public-sha256" in script


def test_db_template_is_explicit_minimum_and_contains_no_credentials() -> None:
    sql = DB_GRANTS_TEMPLATE.read_text(encoding="utf-8")
    upper = sql.upper()
    statements = "\n".join(
        line for line in upper.splitlines() if not line.lstrip().startswith("--")
    )

    assert r"\if :{?strict46_apply}" in sql
    assert r"\quit 42" in sql
    assert "BEGIN;" in sql
    assert "COMMIT;" in sql
    assert "CREATE ROLE" not in statements
    assert "ALTER ROLE" not in statements
    assert "CREATE DATABASE" not in statements
    assert "PASSWORD " not in statements
    assert "GRANT DELETE ON public.scheduler_lease" in sql
    assert upper.count("GRANT DELETE ON") == 1
    assert "REVOKE DELETE, TRUNCATE, REFERENCES, TRIGGER ON TABLE" in sql
    assert "GRANT USAGE, SELECT ON SEQUENCE" in sql
    assert "ALTER DEFAULT PRIVILEGES" not in upper
    assert "GRANT CONNECT ON DATABASE postgres" in sql
    assert "strict46_no_implicit_ownership_assertion" in sql
    assert "REVOKE ALL PRIVILEGES ON TABLE" in sql
    assert "REVOKE ALL PRIVILEGES ON SEQUENCE" in sql
    assert "NOT has_schema_privilege" in sql


def test_bundle_contains_no_private_key_or_secret_assignment() -> None:
    paths = (
        PROVISIONER,
        GUIDE,
        SSH_CONFIG_TEMPLATE,
        KNOWN_HOSTS_TEMPLATE,
        DB_GRANTS_TEMPLATE,
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "BEGIN OPENSSH PRIVATE KEY" not in combined
    assert "BEGIN RSA PRIVATE KEY" not in combined
    assert "BEGIN EC PRIVATE KEY" not in combined
    assert "PG_PASSWORD=" not in combined
    assert "JUDGE_API_KEY=" not in combined


def test_guide_documents_the_closed_independent_runtime_contract() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    assert "/opt/rwkv-strict46/runtime/<manifest_sha256>" in guide
    assert "/var/lib/rwkv-strict46/<run-id>" in guide
    assert "independent system UID" in guide
    assert "content-addressed" in guide
    assert "handoff-requests" in guide
    assert "root-owned system orchestrator" in guide
    assert "Publication is an exact allowlist copy" in guide
    assert "`.env`" in guide
    assert "/home/rwkv/chase/rwkv-skills" not in guide
    assert "systemctl --user" not in guide
    assert "systemd-run --user" not in guide


def test_provisioning_bundle_is_manifest_bound_and_legacy_chains_are_excluded() -> None:
    inventory = set(gate.protocol_inventory_paths(REPO))
    for relative in (
        "ops/g1i_strict46/provision_root_runtime.sh",
        "ops/g1i_strict46/ROOT_PROVISIONING.md",
        "ops/g1i_strict46/templates/ssh_config.157-to-8222.in",
        "ops/g1i_strict46/templates/known_hosts.157-to-8222.in",
        "ops/g1i_strict46/templates/strict46_db_grants.sql.in",
    ):
        assert relative in inventory
    for relative in (
        "ops/g1i_strict46/chain_157_g1h_15_29.sh",
        "ops/g1i_strict46/chain_8222_g1h_133_72.sh",
        "ops/g1i_strict46/wait_and_recover_math.sh",
    ):
        assert relative not in inventory

    banned = (
        "systemctl --user",
        "systemd-run --user",
        "/home/rwkv/chase/rwkv-skills",
        "RWKV_STRICT_CONTROL_REPO",
    )
    for relative in gate.OPS_PATHS:
        text = (REPO / relative).read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{relative} contains {needle}"


def _legacy_guide_does_not_overclaim_current_launcher_or_waiters() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    assert "不得启用生产" in guide
    assert "/home/rwkv/chase/rwkv-skills" in guide
    assert "systemctl --user" in guide
    assert "独立 service UID" in guide
    assert "不含 `.env`" in guide
    assert "行级隔离" in guide
