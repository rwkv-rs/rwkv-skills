#!/usr/bin/env bash
set -euo pipefail

# Root-side, deliberately non-automatic provisioning for the G1i strict-46
# scheduler trust boundary.  This script never creates credentials, grants
# database privileges, enables services, starts a scheduler, or touches a GPU.

readonly PROVISION_SCHEMA="rwkv.g1i-strict46-root-provision.v1"
readonly APPLY_ACK="I_UNDERSTAND_STRICT46_ROOT_PROVISIONING"
readonly SERVICE_USER="rwkv-strict46"
readonly SERVICE_GROUP="rwkv-strict46"
readonly OPT_ROOT="/opt/rwkv-strict46"
readonly RUNTIME_PARENT="$OPT_ROOT/runtime"
readonly PYTHON_PARENT="$OPT_ROOT/python"
readonly INCOMING_ROOT="$OPT_ROOT/incoming"
readonly ETC_ROOT="/etc/rwkv-strict46"
readonly STATE_ROOT="/var/lib/rwkv-strict46"
readonly LOG_ROOT="/var/log/rwkv-strict46"
readonly ROLE_FILE="$ETC_ROOT/host-role"
readonly SSH_ALIAS="strict46-8222"
readonly SSH_CONFIG="$ETC_ROOT/ssh_8222_config"
readonly SSH_IDENTITY="$ETC_ROOT/id_ed25519_8222"
readonly SSH_KNOWN_HOSTS="$ETC_ROOT/known_hosts_8222"
readonly BOOTSTRAP_PYTHON="/usr/bin/python3"
readonly BUILDER_REL="ops/g1i_strict46/build_frozen_runtime.py"
readonly GATE_REL="ops/g1i_strict46/require_global_protocol_gate.py"
readonly LOCK_REL="ops/g1i_strict46/protocol_gate.lock.json"

usage() {
  cat <<'EOF'
Usage:
  provision_root_runtime.sh plan --host-role 157|8222
  provision_root_runtime.sh prepare-host --host-role 157|8222 --apply --ack ACK
  provision_root_runtime.sh inventory-python --python-runtime-id ID
  provision_root_runtime.sh seal-python --host-role ROLE --python-runtime-id ID \
    --python-executable-relative REL --expected-python-inventory-sha256 SHA \
    --apply --ack ACK
  provision_root_runtime.sh publish-runtime --host-role ROLE --repo ABS \
    --approval ABS --approval-sha256 SHA --lock ABS --lock-sha256 SHA \
    --python-runtime-id ID --python-executable-relative REL \
    --expected-python-inventory-sha256 SHA \
    --expected-manifest-sha256 SHA --apply --ack ACK
  provision_root_runtime.sh install-ssh-metadata --host-role 157 \
    --ssh-config-source ABS --expected-ssh-config-sha256 SHA \
    --ssh-known-hosts-source ABS --expected-ssh-known-hosts-sha256 SHA \
    --expected-identity-public-sha256 SHA --apply --ack ACK
  provision_root_runtime.sh verify --host-role ROLE \
    --python-runtime-id ID --python-executable-relative REL \
    --expected-python-inventory-sha256 SHA \
    --expected-manifest-sha256 SHA \
    [157-only SSH SHA arguments from install-ssh-metadata]

ACK is the literal:
  I_UNDERSTAND_STRICT46_ROOT_PROVISIONING

Every mutating phase requires root, --apply, and the acknowledgement.  With no
phase or with incomplete evidence the script exits without changing anything.
It never accepts a password, token, private-key source, or secret value.
EOF
}

die() {
  local code=$1
  shift
  printf 'strict46 provisioning refused: %s\n' "$*" >&2
  exit "$code"
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || die 42 "required command is missing: $1"
}

is_sha256() {
  [[ ${1:-} =~ ^[0-9a-f]{64}$ ]]
}

is_safe_id() {
  [[ ${1:-} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]]
}

is_safe_relative_path() {
  local value=${1:-}
  [[ -n "$value" && "$value" != /* && "$value" != *".."* && "$value" != *$'\n'* ]]
}

require_root() {
  [[ $(id -u) -eq 0 ]] || die 42 "this phase requires root"
}

require_apply_ack() {
  require_root
  [[ $apply_requested == 1 ]] || die 42 "mutating phase requires --apply"
  [[ $acknowledgement == "$APPLY_ACK" ]] || die 42 "acknowledgement mismatch"
}

require_sha() {
  local label=$1
  local value=$2
  is_sha256 "$value" || die 64 "$label must be a lowercase SHA-256"
}

require_absolute_regular_file() {
  local label=$1
  local path=$2
  [[ "$path" == /* ]] || die 64 "$label must be an absolute path"
  [[ ! -L "$path" ]] || die 42 "$label must not be a symlink: $path"
  [[ -f "$path" ]] || die 42 "$label is not a regular file: $path"
}

stable_file_sha256() {
  "$BOOTSTRAP_PYTHON" -I - "$1" <<'PY'
import hashlib
import os
from pathlib import Path
import stat
import sys

path = Path(sys.argv[1])
if path.is_symlink():
    raise SystemExit("symlink refused")
fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
try:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("non-regular file refused")
    digest = hashlib.sha256()
    while chunk := os.read(fd, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(fd)
finally:
    os.close(fd)
current = os.stat(path, follow_symlinks=False)
identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
if identity(before) != identity(after) or identity(after) != identity(current):
    raise SystemExit("file changed while hashing")
print(digest.hexdigest())
PY
}

# Path-independent inventory used to compare independently prepared Python
# environments before either is accepted into a manifest.  Only the executable
# bit is retained because seal-python deliberately normalizes all other modes.
tree_inventory_sha() {
  "$BOOTSTRAP_PYTHON" -I - "$1" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

root_arg = Path(sys.argv[1])
if root_arg.is_symlink():
    raise SystemExit("tree root is a symlink")
root = root_arg.resolve(strict=True)
if not root.is_dir():
    raise SystemExit("tree root is not a directory")

def stable_descriptor(path: Path) -> dict[str, object]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(f"non-regular entry refused: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    current = os.stat(path, follow_symlinks=False)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise SystemExit(f"entry changed while hashing: {path}")
    return {
        "kind": "file",
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
        "executable": bool(after.st_mode & 0o111),
    }

entries: dict[str, dict[str, object]] = {}
for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
    current_path = Path(current)
    for name in sorted(directory_names):
        path = current_path / name
        status = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(status.st_mode):
            raise SystemExit(f"symlink refused: {relative}")
        if not stat.S_ISDIR(status.st_mode):
            raise SystemExit(f"non-directory entry in directory list: {relative}")
        entries[relative] = {"kind": "directory"}
    for name in sorted(file_names):
        path = current_path / name
        status = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(status.st_mode):
            raise SystemExit(f"symlink refused: {relative}")
        if not stat.S_ISREG(status.st_mode):
            raise SystemExit(f"non-regular entry refused: {relative}")
        entries[relative] = stable_descriptor(path)

payload = json.dumps(
    {"schema_version": "rwkv.strict46-python-portable-inventory.v1", "entries": entries},
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
}

assert_root_owned_source_tree() {
  "$BOOTSTRAP_PYTHON" -I - "$1" <<'PY'
import os
from pathlib import Path
import stat
import sys

root_arg = Path(sys.argv[1])
if root_arg.is_symlink():
    raise SystemExit("Python tree root is a symlink")
root = root_arg.resolve(strict=True)
if not root.is_dir():
    raise SystemExit("Python tree root is not a directory")
root_device = root.stat().st_dev
for ancestor in (root.parent, *root.parent.parents):
    status = ancestor.stat()
    if status.st_uid != 0 or status.st_mode & 0o022:
        raise SystemExit(f"untrusted Python ancestor: {ancestor}")
for path in (root, *sorted(root.rglob("*"))):
    status = os.lstat(path)
    if stat.S_ISLNK(status.st_mode):
        raise SystemExit(f"symlink refused: {path}")
    if not (stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode)):
        raise SystemExit(f"non-regular Python entry refused: {path}")
    if status.st_dev != root_device:
        raise SystemExit(f"cross-filesystem entry refused: {path}")
    if status.st_uid != 0 or status.st_mode & 0o022:
        raise SystemExit(f"Python entry is not root-owned/trusted: {path}")
PY
}

assert_sealed_tree() {
  "$BOOTSTRAP_PYTHON" -I - "$1" "$2" <<'PY'
import os
from pathlib import Path
import stat
import sys

root_arg = Path(sys.argv[1])
if root_arg.is_symlink():
    raise SystemExit(f"sealed-tree root is a symlink: {root_arg}")
root = root_arg.resolve(strict=True)
expected_gid = int(sys.argv[2])
root_device = root.stat().st_dev
for ancestor in (root.parent, *root.parent.parents):
    status = ancestor.stat()
    if status.st_uid != 0 or status.st_mode & 0o022:
        raise SystemExit(f"untrusted sealed-tree ancestor: {ancestor}")
for path in (root, *sorted(root.rglob("*"))):
    status = os.lstat(path)
    if stat.S_ISLNK(status.st_mode):
        raise SystemExit(f"symlink in sealed tree: {path}")
    if status.st_dev != root_device:
        raise SystemExit(f"cross-filesystem entry in sealed tree: {path}")
    if status.st_uid != 0 or status.st_gid != expected_gid:
        raise SystemExit(f"sealed-tree owner mismatch: {path}")
    mode = stat.S_IMODE(status.st_mode)
    if stat.S_ISDIR(status.st_mode):
        expected = 0o550
    elif stat.S_ISREG(status.st_mode):
        expected = 0o550 if mode & 0o111 else 0o440
    else:
        raise SystemExit(f"non-regular entry in sealed tree: {path}")
    if mode != expected:
        raise SystemExit(f"sealed-tree mode mismatch: {path}: {mode:o} != {expected:o}")
PY
}

scan_runtime_for_secret_paths() {
  "$BOOTSTRAP_PYTHON" -I - "$1" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
manifest = json.loads((root / "strict46-frozen-runtime.json").read_text(encoding="utf-8"))
blocked_exact = {".env", ".pgpass", ".netrc", "id_rsa", "id_ed25519"}
blocked_suffixes = (".key", ".pem", ".p12", ".pfx")

def blocked(relative: str) -> bool:
    parts = Path(relative).parts
    return any(
        part in blocked_exact
        or part.startswith(".env.")
        or part.endswith(blocked_suffixes)
        for part in parts
    )

bad_paths = []
for path in root.rglob("*"):
    relative = path.relative_to(root).as_posix()
    if blocked(relative):
        bad_paths.append(relative)
for descriptor in manifest.get("support_files", []):
    if isinstance(descriptor, dict) and blocked(str(descriptor.get("path", ""))):
        bad_paths.append(str(descriptor.get("path", "")))
if bad_paths:
    raise SystemExit("secret-bearing path refused: " + ", ".join(sorted(set(bad_paths))))
PY
}

require_prepared_role() {
  local requested_role=$1 gid
  [[ "$requested_role" == 157 || "$requested_role" == 8222 ]] || die 64 "host role must be 157 or 8222"
  [[ -f "$ROLE_FILE" && ! -L "$ROLE_FILE" ]] || die 42 "host has not been prepared"
  gid=$(getent group "$SERVICE_GROUP" | cut -d: -f3)
  assert_root_owned_transport_file "$ROLE_FILE" "$gid" 0440
  [[ $(<"$ROLE_FILE") == "$requested_role" ]] || die 42 "host-role marker mismatch"
}

verify_service_identity() {
  local group_entry user_entry gid service_uid user_gid user_home user_shell other
  group_entry=$(getent group "$SERVICE_GROUP") || die 42 "service group is absent"
  user_entry=$(getent passwd "$SERVICE_USER") || die 42 "service user is absent"
  gid=$(cut -d: -f3 <<<"$group_entry")
  user_gid=$(cut -d: -f4 <<<"$user_entry")
  user_home=$(cut -d: -f6 <<<"$user_entry")
  user_shell=$(cut -d: -f7 <<<"$user_entry")
  [[ "$user_gid" == "$gid" ]] || die 42 "service user primary group mismatch"
  [[ "$user_home" == "$STATE_ROOT" ]] || die 42 "service user home mismatch"
  [[ "$user_shell" == */nologin ]] || die 42 "service user must use nologin"
  service_uid=$(id -u "$SERVICE_USER")
  [[ "$service_uid" -ne 0 ]] || die 42 "service user must not be root"
  for other in rwkv chase; do
    if id -u "$other" >/dev/null 2>&1 && [[ $(id -u "$other") == "$service_uid" ]]; then
      die 42 "service UID is shared with $other"
    fi
  done
}

ensure_directory_once() {
  local path=$1 owner=$2 group=$3 mode=$4 actual
  if [[ -e "$path" || -L "$path" ]]; then
    [[ -d "$path" && ! -L "$path" ]] || die 42 "existing directory target is not a real directory: $path"
    actual=$(stat -c '%U:%G:%a' "$path")
    [[ "$actual" == "$owner:$group:$mode" ]] || \
      die 42 "existing directory metadata differs: $path expected=$owner:$group:$mode actual=$actual"
    return 0
  fi
  install -d -o "$owner" -g "$group" -m "$mode" "$path"
}

prepare_host() {
  local role=$1 nologin group_entry user_entry gid user_gid user_home user_shell tmp
  require_apply_ack
  [[ "$role" == 157 || "$role" == 8222 ]] || die 64 "host role must be 157 or 8222"
  need_command getent
  need_command groupadd
  need_command useradd
  nologin=$(command -v nologin || true)
  [[ -n "$nologin" ]] || die 42 "nologin shell is unavailable"

  if group_entry=$(getent group "$SERVICE_GROUP"); then
    [[ -n "$group_entry" ]] || die 42 "invalid existing service group"
  else
    groupadd --system "$SERVICE_GROUP"
  fi
  gid=$(getent group "$SERVICE_GROUP" | cut -d: -f3)
  if user_entry=$(getent passwd "$SERVICE_USER"); then
    user_gid=$(cut -d: -f4 <<<"$user_entry")
    user_home=$(cut -d: -f6 <<<"$user_entry")
    user_shell=$(cut -d: -f7 <<<"$user_entry")
    [[ "$user_gid" == "$gid" && "$user_home" == "$STATE_ROOT" && "$user_shell" == */nologin ]] || \
      die 42 "existing service account attributes differ; refusing to modify it"
  else
    useradd --system --gid "$SERVICE_GROUP" --home-dir "$STATE_ROOT" \
      --shell "$nologin" --no-create-home "$SERVICE_USER"
  fi
  verify_service_identity

  ensure_directory_once "$OPT_ROOT" root root 755
  ensure_directory_once "$RUNTIME_PARENT" root "$SERVICE_GROUP" 550
  ensure_directory_once "$PYTHON_PARENT" root "$SERVICE_GROUP" 550
  ensure_directory_once "$INCOMING_ROOT" root root 700
  ensure_directory_once "$ETC_ROOT" root "$SERVICE_GROUP" 550
  # The service UID owns only this fixed parent.  Each launcher creates one
  # private /var/lib/rwkv-strict46/<run-id> tree through runtime_state.py;
  # mutable state is never written back into a development checkout.
  ensure_directory_once "$STATE_ROOT" "$SERVICE_USER" "$SERVICE_GROUP" 700
  ensure_directory_once "$LOG_ROOT" "$SERVICE_USER" "$SERVICE_GROUP" 750

  if [[ -e "$ROLE_FILE" || -L "$ROLE_FILE" ]]; then
    [[ ! -L "$ROLE_FILE" && -f "$ROLE_FILE" && $(<"$ROLE_FILE") == "$role" ]] || \
      die 42 "existing host-role marker differs; refusing overwrite"
    assert_root_owned_transport_file "$ROLE_FILE" "$gid" 0440
  else
    tmp=$(mktemp "$ETC_ROOT/.host-role.XXXXXXXX")
    printf '%s\n' "$role" >"$tmp"
    chown root:"$SERVICE_GROUP" "$tmp"
    chmod 0440 "$tmp"
    mv -T "$tmp" "$ROLE_FILE"
  fi
  printf 'prepared host role=%s service_uid=%s schema=%s\n' \
    "$role" "$(id -u "$SERVICE_USER")" "$PROVISION_SCHEMA"
}

seal_python() {
  local role=$1 runtime_id=$2 executable_relative=$3 expected=$4 root actual gid
  require_apply_ack
  require_prepared_role "$role"
  is_safe_id "$runtime_id" || die 64 "unsafe Python runtime id"
  is_safe_relative_path "$executable_relative" || die 64 "unsafe Python executable relative path"
  require_sha "expected Python inventory" "$expected"
  root="$PYTHON_PARENT/$runtime_id"
  [[ -d "$root" && ! -L "$root" ]] || die 42 "Python runtime must already exist at $root"
  [[ -f "$root/$executable_relative" && ! -L "$root/$executable_relative" && -x "$root/$executable_relative" ]] || \
    die 42 "Python executable is absent, linked, or not executable"
  assert_root_owned_source_tree "$root"
  actual=$(tree_inventory_sha "$root")
  [[ "$actual" == "$expected" ]] || die 42 "Python inventory mismatch: expected=$expected actual=$actual"
  gid=$(getent group "$SERVICE_GROUP" | cut -d: -f3)
  chown -R root:"$SERVICE_GROUP" "$root"
  find "$root" -type f -perm /111 -exec chmod 0550 {} +
  find "$root" -type f ! -perm /111 -exec chmod 0440 {} +
  find "$root" -type d -exec chmod 0550 {} +
  assert_sealed_tree "$root" "$gid"
  actual=$(tree_inventory_sha "$root")
  [[ "$actual" == "$expected" ]] || die 42 "Python inventory changed while sealing"
  printf 'sealed python=%s inventory_sha256=%s\n' "$root" "$actual"
}

runtime_approval_path() {
  "$BOOTSTRAP_PYTHON" -I - "$1" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
manifest = json.loads((root / "strict46-frozen-runtime.json").read_text(encoding="utf-8"))
relative = Path(str(manifest.get("approval", {}).get("path", "")))
if relative.is_absolute() or ".." in relative.parts or not relative.parts:
    raise SystemExit("invalid frozen approval path")
candidate = (root / relative).resolve(strict=True)
candidate.relative_to(root)
print(candidate)
PY
}

verify_runtime() {
  local root=$1 expected=$2 approval gate lock gid
  [[ -d "$root" && ! -L "$root" ]] || die 42 "runtime is missing or linked: $root"
  [[ $(basename "$root") == "$expected" ]] || die 42 "runtime directory is not the expected manifest SHA"
  gid=$(getent group "$SERVICE_GROUP" | cut -d: -f3)
  assert_sealed_tree "$root" "$gid"
  scan_runtime_for_secret_paths "$root"
  approval=$(runtime_approval_path "$root")
  gate="$root/$GATE_REL"
  lock="$root/$LOCK_REL"
  env PYTHONDONTWRITEBYTECODE=1 "$BOOTSTRAP_PYTHON" -B -I "$gate" \
    --repo "$root" --lock "$lock" --approval "$approval" \
    --frozen-runtime "$root" --print-frozen-python >/dev/null
}

cleanup_publish_stage() {
  local stage=${publish_stage:-}
  if [[ -n "$stage" && "$stage" == "$INCOMING_ROOT"/publish.* && -d "$stage" && ! -L "$stage" ]]; then
    rm -rf --one-file-system -- "$stage"
  fi
}

publish_runtime() {
  local role=$1 repo=$2 approval=$3 approval_sha=$4 lock=$5 lock_sha=$6
  local runtime_id=$7 executable_relative=$8 python_inventory=$9 expected_manifest=${10}
  local python_root target actual output_parent built_output gid
  require_apply_ack
  require_prepared_role "$role"
  is_safe_id "$runtime_id" || die 64 "unsafe Python runtime id"
  is_safe_relative_path "$executable_relative" || die 64 "unsafe Python executable relative path"
  require_sha "approval SHA" "$approval_sha"
  require_sha "lock SHA" "$lock_sha"
  require_sha "expected Python inventory" "$python_inventory"
  require_sha "expected manifest" "$expected_manifest"
  [[ "$repo" == /* && -d "$repo" && ! -L "$repo" ]] || die 42 "repo must be an absolute non-symlink directory"
  assert_root_owned_source_tree "$repo"
  require_absolute_regular_file "approval" "$approval"
  require_absolute_regular_file "lock" "$lock"
  case "$(readlink -f -- "$approval")" in "$repo"/*) ;; *) die 42 "approval is outside release repo" ;; esac
  case "$(readlink -f -- "$lock")" in "$repo"/*) ;; *) die 42 "lock is outside release repo" ;; esac
  [[ $(stable_file_sha256 "$approval") == "$approval_sha" ]] || die 42 "approval SHA mismatch"
  [[ $(stable_file_sha256 "$lock") == "$lock_sha" ]] || die 42 "lock SHA mismatch"
  [[ ! -e "$repo/.env" && ! -L "$repo/.env" ]] || die 42 "repo/.env is forbidden in a published runtime"
  [[ -f "$repo/$BUILDER_REL" && -f "$repo/$GATE_REL" ]] || die 42 "repo lacks strict46 publisher/gate"

  python_root="$PYTHON_PARENT/$runtime_id"
  gid=$(getent group "$SERVICE_GROUP" | cut -d: -f3)
  assert_sealed_tree "$python_root" "$gid"
  actual=$(tree_inventory_sha "$python_root")
  [[ "$actual" == "$python_inventory" ]] || die 42 "sealed Python inventory mismatch"
  [[ -f "$python_root/$executable_relative" && -x "$python_root/$executable_relative" && ! -L "$python_root/$executable_relative" ]] || \
    die 42 "sealed Python executable mismatch"

  target="$RUNTIME_PARENT/$expected_manifest"
  if [[ -e "$target" || -L "$target" ]]; then
    verify_runtime "$target" "$expected_manifest"
    printf 'runtime already published and verified: %s\n' "$target"
    return 0
  fi

  publish_stage=$(mktemp -d "$INCOMING_ROOT/publish.XXXXXXXX")
  trap cleanup_publish_stage EXIT
  output_parent="$publish_stage/output"
  install -d -o root -g root -m 0700 "$output_parent"
  built_output=$(
    env PYTHONDONTWRITEBYTECODE=1 "$python_root/$executable_relative" -B -I "$repo/$BUILDER_REL" \
      --repo "$repo" --approval "$approval" --lock "$lock" \
      --output-parent "$output_parent" --runtime-gid "$gid" \
      --python-runtime-root "$python_root" \
      --python-executable "$python_root/$executable_relative"
  )
  [[ "$built_output" == "$output_parent"/* && -d "$built_output" && ! -L "$built_output" ]] || \
    die 42 "publisher returned an unexpected output path"
  [[ $(basename "$built_output") == "$expected_manifest" ]] || \
    die 42 "publisher manifest differs: expected=$expected_manifest actual=$(basename "$built_output")"
  [[ $(stable_file_sha256 "$approval") == "$approval_sha" ]] || die 42 "approval changed during publication"
  [[ $(stable_file_sha256 "$lock") == "$lock_sha" ]] || die 42 "lock changed during publication"
  verify_runtime "$built_output" "$expected_manifest"
  [[ ! -e "$target" && ! -L "$target" ]] || die 42 "runtime target appeared during publication"
  mv -T -n -- "$built_output" "$target"
  [[ ! -e "$built_output" && ! -L "$built_output" ]] || die 42 "atomic no-clobber publication lost a race"
  verify_runtime "$target" "$expected_manifest"
  cleanup_publish_stage
  trap - EXIT
  printf 'published runtime=%s manifest_sha256=%s\n' "$target" "$expected_manifest"
}

assert_root_owned_transport_file() {
  "$BOOTSTRAP_PYTHON" -I - "$1" "$2" "$3" <<'PY'
import os
from pathlib import Path
import stat
import sys

path = Path(sys.argv[1])
expected_gid = int(sys.argv[2])
expected_mode = int(sys.argv[3], 8)
if path.is_symlink():
    raise SystemExit(f"transport file is a symlink: {path}")
resolved = path.resolve(strict=True)
status = resolved.stat()
if not stat.S_ISREG(status.st_mode):
    raise SystemExit(f"transport path is not a regular file: {resolved}")
if status.st_uid != 0 or status.st_gid != expected_gid:
    raise SystemExit(f"transport owner mismatch: {resolved}")
if stat.S_IMODE(status.st_mode) != expected_mode:
    raise SystemExit(f"transport mode mismatch: {resolved}")
for ancestor in (resolved.parent, *resolved.parent.parents):
    status = ancestor.stat()
    if status.st_uid != 0 or status.st_mode & 0o022:
        raise SystemExit(f"untrusted transport ancestor: {ancestor}")
PY
}

assert_root_owned_readonly_input() {
  "$BOOTSTRAP_PYTHON" -I - "$1" <<'PY'
import os
from pathlib import Path
import stat
import sys

path = Path(sys.argv[1])
if path.is_symlink():
    raise SystemExit(f"input is a symlink: {path}")
resolved = path.resolve(strict=True)
status = resolved.stat()
if not stat.S_ISREG(status.st_mode) or status.st_uid != 0 or status.st_mode & 0o022:
    raise SystemExit(f"input must be a root-owned, non-group-writable regular file: {resolved}")
for ancestor in (resolved.parent, *resolved.parent.parents):
    status = ancestor.stat()
    if status.st_uid != 0 or status.st_mode & 0o022:
        raise SystemExit(f"untrusted input ancestor: {ancestor}")
PY
}

reject_placeholders() {
  if grep -Eq '__STRICT46_|CHANGEME|PLACEHOLDER|BEGIN (OPENSSH |RSA |EC )?PRIVATE KEY' "$1"; then
    die 42 "placeholder or private-key material refused in public metadata: $1"
  fi
}

assert_ssh_metadata() {
  local config_sha=$1 known_hosts_sha=$2 identity_public_sha=$3 gid expanded value hostname port lookup
  require_sha "expected SSH config SHA" "$config_sha"
  require_sha "expected SSH known_hosts SHA" "$known_hosts_sha"
  require_sha "expected identity public SHA" "$identity_public_sha"
  gid=$(getent group "$SERVICE_GROUP" | cut -d: -f3)
  assert_root_owned_transport_file "$SSH_CONFIG" "$gid" 0440
  assert_root_owned_transport_file "$SSH_KNOWN_HOSTS" "$gid" 0440
  assert_root_owned_transport_file "$SSH_IDENTITY" "$gid" 0440
  [[ $(stable_file_sha256 "$SSH_CONFIG") == "$config_sha" ]] || die 42 "installed SSH config SHA mismatch"
  [[ $(stable_file_sha256 "$SSH_KNOWN_HOSTS") == "$known_hosts_sha" ]] || die 42 "installed known_hosts SHA mismatch"
  reject_placeholders "$SSH_CONFIG"
  reject_placeholders "$SSH_KNOWN_HOSTS"
  value=$(/usr/bin/ssh-keygen -y -f "$SSH_IDENTITY" </dev/null | sha256sum | awk '{print $1}') || \
    die 42 "identity must be an unencrypted, valid private key"
  [[ "$value" == "$identity_public_sha" ]] || die 42 "identity public-key SHA mismatch"
  grep -Eiq '^[[:space:]]*Host[[:space:]]+strict46-8222[[:space:]]*$' "$SSH_CONFIG" || \
    die 42 "SSH config lacks the exact strict46-8222 host stanza"
  if grep -Eiq '^[[:space:]]*(Include|Match|LocalForward|RemoteForward|DynamicForward)[[:space:]]' "$SSH_CONFIG"; then
    die 42 "SSH config contains an unapproved dynamic/include/forward directive"
  fi
  expanded=$(/usr/bin/ssh -G -F "$SSH_CONFIG" "$SSH_ALIAS" 2>/dev/null) || die 42 "ssh -G rejected installed config"
  get_value() { awk -v key="$1" '$1 == key {print $2; exit}' <<<"$expanded"; }
  [[ $(get_value identityfile) == "$SSH_IDENTITY" ]] || die 42 "SSH IdentityFile is not pinned"
  [[ $(get_value userknownhostsfile) == "$SSH_KNOWN_HOSTS" ]] || die 42 "SSH UserKnownHostsFile is not pinned"
  [[ $(get_value globalknownhostsfile) == /dev/null ]] || die 42 "SSH global known_hosts is not disabled"
  for value in \
    "batchmode yes" "identitiesonly yes" "stricthostkeychecking true" \
    "passwordauthentication no" "kbdinteractiveauthentication no" \
    "forwardagent no" "controlmaster false" "permitlocalcommand no"; do
    [[ $(get_value "${value%% *}") == "${value#* }" ]] || die 42 "SSH option is not fail-closed: $value"
  done
  value=$(get_value proxycommand)
  [[ -z "$value" || "$value" == none ]] || die 42 "SSH ProxyCommand is forbidden"
  value=$(get_value proxyjump)
  [[ -z "$value" || "$value" == none ]] || die 42 "SSH ProxyJump is forbidden"
  hostname=$(get_value hostname)
  port=$(get_value port)
  [[ -n "$hostname" && "$hostname" != *"__"* && "$port" =~ ^[0-9]+$ ]] || die 42 "SSH destination is invalid"
  lookup=$hostname
  [[ "$port" == 22 ]] || lookup="[$hostname]:$port"
  /usr/bin/ssh-keygen -F "$lookup" -f "$SSH_KNOWN_HOSTS" >/dev/null || \
    die 42 "known_hosts does not bind the configured destination"
}

install_public_file_once() {
  local source=$1 destination=$2 expected=$3 gid=$4 tmp
  if [[ -e "$destination" || -L "$destination" ]]; then
    [[ ! -L "$destination" && -f "$destination" ]] || die 42 "existing destination is not a regular file"
    [[ $(stable_file_sha256 "$destination") == "$expected" ]] || die 42 "existing destination differs: $destination"
    assert_root_owned_transport_file "$destination" "$gid" 0440
    return 0
  fi
  tmp=$(mktemp "$ETC_ROOT/.install.XXXXXXXX")
  install -o root -g "$SERVICE_GROUP" -m 0440 "$source" "$tmp"
  [[ $(stable_file_sha256 "$tmp") == "$expected" ]] || die 42 "installed copy SHA mismatch"
  mv -T -n -- "$tmp" "$destination"
  [[ ! -e "$tmp" && ! -L "$tmp" ]] || die 42 "no-clobber metadata install lost a race"
}

install_ssh_metadata() {
  local role=$1 config_source=$2 config_sha=$3 known_source=$4 known_sha=$5 public_sha=$6 gid
  require_apply_ack
  require_prepared_role "$role"
  [[ "$role" == 157 ]] || die 42 "157-to-8222 SSH metadata belongs only on host role 157"
  require_absolute_regular_file "SSH config source" "$config_source"
  require_absolute_regular_file "SSH known_hosts source" "$known_source"
  assert_root_owned_readonly_input "$config_source"
  assert_root_owned_readonly_input "$known_source"
  require_sha "expected SSH config SHA" "$config_sha"
  require_sha "expected SSH known_hosts SHA" "$known_sha"
  require_sha "expected identity public SHA" "$public_sha"
  [[ $(stable_file_sha256 "$config_source") == "$config_sha" ]] || die 42 "SSH config source SHA mismatch"
  [[ $(stable_file_sha256 "$known_source") == "$known_sha" ]] || die 42 "known_hosts source SHA mismatch"
  reject_placeholders "$config_source"
  reject_placeholders "$known_source"
  gid=$(getent group "$SERVICE_GROUP" | cut -d: -f3)
  assert_root_owned_transport_file "$SSH_IDENTITY" "$gid" 0440
  install_public_file_once "$config_source" "$SSH_CONFIG" "$config_sha" "$gid"
  install_public_file_once "$known_source" "$SSH_KNOWN_HOSTS" "$known_sha" "$gid"
  assert_ssh_metadata "$config_sha" "$known_sha" "$public_sha"
  printf 'installed pinned SSH metadata; private key was not created or copied\n'
}

verify_all() {
  local role=$1 runtime_id=$2 executable_relative=$3 python_inventory=$4 manifest=$5
  local config_sha=$6 known_sha=$7 public_sha=$8 python_root runtime_root gid actual
  require_root
  require_prepared_role "$role"
  verify_service_identity
  is_safe_id "$runtime_id" || die 64 "unsafe Python runtime id"
  is_safe_relative_path "$executable_relative" || die 64 "unsafe Python executable relative path"
  require_sha "expected Python inventory" "$python_inventory"
  require_sha "expected manifest" "$manifest"
  gid=$(getent group "$SERVICE_GROUP" | cut -d: -f3)
  python_root="$PYTHON_PARENT/$runtime_id"
  assert_sealed_tree "$python_root" "$gid"
  actual=$(tree_inventory_sha "$python_root")
  [[ "$actual" == "$python_inventory" ]] || die 42 "Python inventory mismatch"
  [[ -x "$python_root/$executable_relative" && ! -L "$python_root/$executable_relative" ]] || die 42 "Python executable mismatch"
  runtime_root="$RUNTIME_PARENT/$manifest"
  verify_runtime "$runtime_root" "$manifest"
  if [[ "$role" == 157 ]]; then
    assert_ssh_metadata "$config_sha" "$known_sha" "$public_sha"
  fi
  printf 'verification passed host=%s runtime=%s python_inventory_sha256=%s\n' \
    "$role" "$runtime_root" "$actual"
}

phase=${1:-}
[[ -n "$phase" ]] || { usage >&2; exit 64; }
shift || true

host_role=""
repo=""
approval=""
approval_sha=""
lock=""
lock_sha=""
python_runtime_id=""
python_executable_relative=""
python_inventory_sha=""
manifest_sha=""
ssh_config_source=""
ssh_config_sha=""
ssh_known_hosts_source=""
ssh_known_hosts_sha=""
identity_public_sha=""
apply_requested=0
acknowledgement=""

while (($#)); do
  if [[ "$1" == --* && "$1" != --apply && "$1" != --help && $# -lt 2 ]]; then
    die 64 "missing value for argument: $1"
  fi
  case "$1" in
    --host-role) host_role=${2:-}; shift 2 ;;
    --repo) repo=${2:-}; shift 2 ;;
    --approval) approval=${2:-}; shift 2 ;;
    --approval-sha256) approval_sha=${2:-}; shift 2 ;;
    --lock) lock=${2:-}; shift 2 ;;
    --lock-sha256) lock_sha=${2:-}; shift 2 ;;
    --python-runtime-id) python_runtime_id=${2:-}; shift 2 ;;
    --python-executable-relative) python_executable_relative=${2:-}; shift 2 ;;
    --expected-python-inventory-sha256) python_inventory_sha=${2:-}; shift 2 ;;
    --expected-manifest-sha256) manifest_sha=${2:-}; shift 2 ;;
    --ssh-config-source) ssh_config_source=${2:-}; shift 2 ;;
    --expected-ssh-config-sha256) ssh_config_sha=${2:-}; shift 2 ;;
    --ssh-known-hosts-source) ssh_known_hosts_source=${2:-}; shift 2 ;;
    --expected-ssh-known-hosts-sha256) ssh_known_hosts_sha=${2:-}; shift 2 ;;
    --expected-identity-public-sha256) identity_public_sha=${2:-}; shift 2 ;;
    --apply) apply_requested=1; shift ;;
    --ack) acknowledgement=${2:-}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die 64 "unknown or incomplete argument: $1" ;;
  esac
done

case "$phase" in
  plan)
    [[ "$host_role" == 157 || "$host_role" == 8222 ]] || die 64 "plan requires --host-role 157|8222"
    cat <<EOF
schema=$PROVISION_SCHEMA
host_role=$host_role
mutations=none
service_identity=$SERVICE_USER:$SERVICE_GROUP
runtime_parent=$RUNTIME_PARENT/<manifest_sha256>
python_parent=$PYTHON_PARENT/<reviewed_id>
db_changes=never_performed_by_this_script
services_started_or_enabled=never
EOF
    if [[ "$host_role" == 157 ]]; then
      printf 'ssh_route=%s -> %s via %s,%s,%s\n' 157 8222 "$SSH_CONFIG" "$SSH_IDENTITY" "$SSH_KNOWN_HOSTS"
    fi
    ;;
  prepare-host)
    prepare_host "$host_role"
    ;;
  inventory-python)
    is_safe_id "$python_runtime_id" || die 64 "inventory-python requires a safe --python-runtime-id"
    tree_inventory_sha "$PYTHON_PARENT/$python_runtime_id"
    ;;
  seal-python)
    seal_python "$host_role" "$python_runtime_id" "$python_executable_relative" "$python_inventory_sha"
    ;;
  publish-runtime)
    publish_runtime "$host_role" "$repo" "$approval" "$approval_sha" "$lock" "$lock_sha" \
      "$python_runtime_id" "$python_executable_relative" "$python_inventory_sha" "$manifest_sha"
    ;;
  install-ssh-metadata)
    install_ssh_metadata "$host_role" "$ssh_config_source" "$ssh_config_sha" \
      "$ssh_known_hosts_source" "$ssh_known_hosts_sha" "$identity_public_sha"
    ;;
  verify)
    verify_all "$host_role" "$python_runtime_id" "$python_executable_relative" \
      "$python_inventory_sha" "$manifest_sha" "$ssh_config_sha" \
      "$ssh_known_hosts_sha" "$identity_public_sha"
    ;;
  *)
    usage >&2
    die 64 "unknown phase: $phase"
    ;;
esac
