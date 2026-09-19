#!/bin/bash
# serve_backend.sh — how the bake-off's SGLang server is started and stopped.
#
# Sourced by run_model_cycle.sh and smoke_kernel.sh (not executed). Two modes,
# chosen by SERVE_MODE in the environment or, when unset, by the tracked file
# evals/swebench/serve_mode.conf — so a mode change lands at the next cycle of
# an already-running queue without relaunching it:
#
#   bare    scripts/launch.sh from the live conda env + SGLANG_DIR (the pre-
#           2026-09 behaviour; the serving stack is whatever the tree holds).
#   docker  the repo's OCI image (Dockerfile -> sglang-cuda-3090:local). Same
#           launch.sh + preset, but the serving stack is the pinned, GPU-free-
#           built image: one immutable artifact per cell, recorded in every
#           lane's meta.json. The image runs secure-launch (file-backed API
#           keys, UID 10001), so this mode mints a per-cycle key pair and
#           exports SWEBENCH_API_KEY_FILE for docker_rollout.py, which hands
#           the key to every scaffold and its own preflight.
#
# Environment (docker mode):
#   SERVE_IMAGE              image ref            (default sglang-cuda-3090:local)
#   SERVE_CONTAINER          container name       (default bakeoff-sglang)
#   SERVE_SECRETS_DIR        key-pair dir         (default $LOG_DIR/secrets)
#   SERVE_CACHE_VOLUME       named volume for /home/sglang/.cache — the Triton /
#                            flashinfer JIT caches (default derived from the image ref)
#   SERVE_TRUST_REMOTE_CODE  SGLANG_TRUST_REMOTE_CODE for the container (default 1:
#                            parity with bare metal, which always passes
#                            --trust-remote-code; the mounted checkpoints are ours)
#   SERVE_ALLOW_STALE_IMAGE  1 = don't fail when the image's scripts/launch.sh
#                            differs from the repo's (default 0: a stale image
#                            would roll a cell under a preset nobody committed)
#   SERVE_GPUS               --gpus value (default all)
# launch.sh knobs (QUANT MEM CTX KV_DTYPE MAX_RUNNING TP DTYPE CHUNKED EXTRA_ARGS DRY_RUN
# OVERRIDE_ARGS ENABLE_CUSTOM_AR CUDA_VISIBLE_DEVICES) are forwarded into the
# container when set. MODEL= overrides must use the in-container path (/models/...).
#
# Functions:
#   serve_backend_init                 resolve mode, mint keys, stale-image check;
#                                      exports SERVE_MODE, SWEBENCH_API_KEY_FILE,
#                                      SWEBENCH_SERVE_MODE/IMAGE/IMAGE_ID
#   serve_start <preset> <log> <pidfile> [launch.sh args...]
#   serve_stop <preset> [port]
#   serve_models [port]                ids on /v1/models (authed in docker mode)
#   serve_curl <url> [curl args...]    curl with the bake-off key when one exists
#   serve_receipt <out.json>           what served this cycle (mode/image/tree)

SERVE_REPO_DIR="${SERVE_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SERVE_MODE_CONF="$SERVE_REPO_DIR/evals/swebench/serve_mode.conf"
SERVE_IMAGE="${SERVE_IMAGE:-sglang-cuda-3090:local}"
SERVE_CONTAINER="${SERVE_CONTAINER:-bakeoff-sglang}"
SERVE_GPUS="${SERVE_GPUS:-all}"
SERVE_TRUST_REMOTE_CODE="${SERVE_TRUST_REMOTE_CODE:-1}"
SERVE_ALLOW_STALE_IMAGE="${SERVE_ALLOW_STALE_IMAGE:-0}"
SERVE_LAUNCH_KNOBS=(QUANT MEM CTX KV_DTYPE MAX_RUNNING TP DTYPE CHUNKED EXTRA_ARGS OVERRIDE_ARGS ENABLE_CUSTOM_AR CUDA_VISIBLE_DEVICES DRY_RUN)

_serve_log() { echo "[serve:${SERVE_MODE:-?} $(date +%H:%M:%S)] $*"; }

_serve_resolve_mode() {
  if [ -z "${SERVE_MODE:-}" ]; then
    SERVE_MODE="$(sed -n '1{s/[[:space:]#].*//;p}' "$SERVE_MODE_CONF" 2>/dev/null)"
    SERVE_MODE="${SERVE_MODE:-bare}"
  fi
  case "$SERVE_MODE" in
    bare|docker) ;;
    *) echo "ERROR: SERVE_MODE must be bare or docker (got '$SERVE_MODE')" >&2; return 2 ;;
  esac
  export SERVE_MODE
}

# Per-cycle key pair. Files are 0644 inside a 0700 dir: secure-launch (UID
# 10001) reads them through a file bind mount, which does not traverse the
# host directory, while other local users are stopped at the directory.
# Existing keys are reused so a relaunched cycle can re-attach to a running
# container.
_serve_mint_keys() {
  SERVE_SECRETS_DIR="${SERVE_SECRETS_DIR:-${LOG_DIR:-/tmp/run-model-cycle-logs}/secrets}"
  mkdir -p "$SERVE_SECRETS_DIR" && chmod 0700 "$SERVE_SECRETS_DIR" || return 1
  local f
  for f in api-key admin-key; do
    if [ ! -s "$SERVE_SECRETS_DIR/$f" ]; then
      (umask 022; python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > "$SERVE_SECRETS_DIR/$f") || return 1
    fi
    chmod 0644 "$SERVE_SECRETS_DIR/$f"
  done
  export SERVE_SECRETS_DIR SWEBENCH_API_KEY_FILE="$SERVE_SECRETS_DIR/api-key"
}

_serve_api_key() {
  [ -n "${SWEBENCH_API_KEY_FILE:-}" ] && [ -s "$SWEBENCH_API_KEY_FILE" ] && head -n1 "$SWEBENCH_API_KEY_FILE"
}

serve_curl() {
  local url="$1"; shift
  local key; key="$(_serve_api_key)"
  if [ -n "$key" ]; then
    curl -s -H "Authorization: Bearer $key" "$@" "$url"
  else
    curl -s "$@" "$url"
  fi
}

serve_models() {
  local port="${1:-23334}"
  serve_curl "http://127.0.0.1:$port/v1/models" -m 5 2>/dev/null \
    | python3 -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null
}

# The image COPYs scripts/launch.sh at build time. A cell must run the preset
# as committed, so the image's copy has to match the repo's — otherwise the
# image needs a rebuild (at a cycle boundary; never while a lane rolls).
_serve_image_check() {
  if ! docker image inspect "$SERVE_IMAGE" >/dev/null 2>&1; then
    echo "ERROR: serve image $SERVE_IMAGE not present (DOCKER_BUILDKIT=1 docker build -t $SERVE_IMAGE .)" >&2
    return 1
  fi
  SERVE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$SERVE_IMAGE")"
  SERVE_IMAGE_LAUNCH_SHA="$(docker run --rm --entrypoint sha256sum "$SERVE_IMAGE" /opt/3090-inference/scripts/launch.sh 2>/dev/null | cut -d' ' -f1)"
  SERVE_REPO_LAUNCH_SHA="$(sha256sum "$SERVE_REPO_DIR/scripts/launch.sh" | cut -d' ' -f1)"
  if [ "$SERVE_IMAGE_LAUNCH_SHA" != "$SERVE_REPO_LAUNCH_SHA" ]; then
    if [ "$SERVE_ALLOW_STALE_IMAGE" = "1" ]; then
      _serve_log "WARNING: image launch.sh ${SERVE_IMAGE_LAUNCH_SHA:0:12} != repo ${SERVE_REPO_LAUNCH_SHA:0:12} (SERVE_ALLOW_STALE_IMAGE=1)"
    else
      echo "ERROR: $SERVE_IMAGE carries scripts/launch.sh ${SERVE_IMAGE_LAUNCH_SHA:-<none>} but the repo has ${SERVE_REPO_LAUNCH_SHA:0:12} — rebuild the image (or SERVE_ALLOW_STALE_IMAGE=1)" >&2
      return 1
    fi
  fi
  export SWEBENCH_SERVE_IMAGE="$SERVE_IMAGE" SWEBENCH_SERVE_IMAGE_ID="$SERVE_IMAGE_ID"
}

serve_backend_init() {
  _serve_resolve_mode || return $?
  export SWEBENCH_SERVE_MODE="$SERVE_MODE"
  if [ "$SERVE_MODE" = "docker" ]; then
    command -v docker >/dev/null || { echo "ERROR: SERVE_MODE=docker but no docker CLI" >&2; return 1; }
    _serve_mint_keys || { echo "ERROR: could not mint the bake-off API key pair" >&2; return 1; }
    _serve_image_check || return 1
    local tag="${SERVE_IMAGE##*/}"
    SERVE_CACHE_VOLUME="${SERVE_CACHE_VOLUME:-sglang-3090-cache-${tag//[^A-Za-z0-9_.-]/-}}"
    _serve_log "image $SERVE_IMAGE (${SERVE_IMAGE_ID#sha256:}) launch.sh ${SERVE_IMAGE_LAUNCH_SHA:0:12}; keys $SERVE_SECRETS_DIR; cache volume $SERVE_CACHE_VOLUME"
  else
    unset SWEBENCH_API_KEY_FILE SWEBENCH_SERVE_IMAGE SWEBENCH_SERVE_IMAGE_ID
  fi
}

_serve_container_state() {  # running | exited | absent
  local st; st="$(docker inspect --format '{{.State.Status}}' "$1" 2>/dev/null)"
  echo "${st:-absent}"
}

_serve_port_busy() {  # 0 when something already listens on tcp:$1
  ss -Hltn "sport = :$1" 2>/dev/null | grep -q .
}

serve_start() {
  local preset="$1" log="$2" pidfile="$3"; shift 3
  if [ "$SERVE_MODE" = "bare" ]; then
    # 9>&-: don't leak run_all_cycles.sh's flock fd into the server — a server
    # that outlives a stopped queue would otherwise hold /tmp/swebench-bakeoff.lock
    # and block the relaunch (2026-09-13).
    nohup setsid bash "$SERVE_REPO_DIR/scripts/launch.sh" "$preset" "$@" \
      > "$log" 2>&1 < /dev/null 9>&- &
    local pid=$!
    disown $pid 2>/dev/null
    echo $pid > "$pidfile"
    return 0
  fi

  local port=23334 i
  for ((i = 1; i <= $#; i++)); do [ "${!i}" = "--port" ] && { local j=$((i + 1)); port="${!j}"; }; done
  local name="$SERVE_CONTAINER"; [ "$port" != "23334" ] && name="$SERVE_CONTAINER-$port"
  docker rm -f "$name" >/dev/null 2>&1 || true
  if _serve_port_busy "$port"; then
    echo "ERROR: tcp:$port already has a listener (a bare-metal server or another container) — stop it before serving from the image" >&2
    ss -Hltnp "sport = :$port" 2>/dev/null >&2
    return 1
  fi

  local models; models="$(readlink -f "$MODELS_DIR")"
  local args=(
    docker run -d --name "$name"
    --gpus "$SERVE_GPUS" --network=host --shm-size 16g
    --cap-drop=ALL --security-opt=no-new-privileges:true --pids-limit 4096
    --stop-timeout 90
    -e SGLANG_API_KEY_FILE=/run/secrets/sglang-api-key
    -e SGLANG_ADMIN_API_KEY_FILE=/run/secrets/sglang-admin-api-key
    -e SGLANG_TRUST_REMOTE_CODE="$SERVE_TRUST_REMOTE_CODE"
    -e SGLANG_ENABLE_METRICS=1
    --mount "type=bind,src=$SERVE_SECRETS_DIR/api-key,dst=/run/secrets/sglang-api-key,readonly"
    --mount "type=bind,src=$SERVE_SECRETS_DIR/admin-key,dst=/run/secrets/sglang-admin-api-key,readonly"
    --mount "type=bind,src=$models,dst=/models,readonly"
    --mount "type=volume,src=$SERVE_CACHE_VOLUME,dst=/home/sglang/.cache"
  )
  local k
  for k in "${SERVE_LAUNCH_KNOBS[@]}"; do [ -n "${!k+x}" ] && args+=(-e "$k"); done
  args+=("$SERVE_IMAGE" scripts/launch.sh "$preset" "$@")
  local cid
  if ! cid="$("${args[@]}" 9>&-)"; then
    echo "ERROR: docker run failed for $name" >&2
    return 1
  fi
  echo "$cid" > "${pidfile%.pid}.cid"
  # keep a plain server.log for wait_ready's tail, the Phase-4 grep and humans
  nohup setsid docker logs -f "$name" > "$log" 2>&1 < /dev/null 9>&- &
  local lp=$!
  disown $lp 2>/dev/null
  echo $lp > "$pidfile"
  _serve_log "container $name ${cid:0:12} started (log follower pid $lp)"
}

serve_stop() {
  local preset="$1" port="${2:-23334}"
  if [ "$SERVE_MODE" = "bare" ]; then
    if [ "$port" = "23334" ]; then
      pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
      pkill -KILL -f "scripts/launch.sh $preset" 2>/dev/null || true
    else  # side ports (kernel smoke): only that server's process tree
      pkill -KILL -f "sglang.launch_server.*--port $port" 2>/dev/null || true
      pkill -KILL -f "scripts/launch.sh $preset --port $port" 2>/dev/null || true
    fi
    sleep 5
    return 0
  fi
  local name="$SERVE_CONTAINER"; [ "$port" != "23334" ] && name="$SERVE_CONTAINER-$port"
  if [ "$(_serve_container_state "$name")" != "absent" ]; then
    docker stop "$name" >/dev/null 2>&1 || true   # SIGTERM, --stop-timeout, then KILL
    docker rm -f "$name" >/dev/null 2>&1 || true
  fi
  sleep 2
}

# True when the named container is up and serves $2 — a cycle relaunched
# after a harness-side fix re-attaches instead of fighting for the port.
serve_reusable() {
  local served="$1" port="${2:-23334}"
  if [ "$SERVE_MODE" = "docker" ]; then
    local name="$SERVE_CONTAINER"; [ "$port" != "23334" ] && name="$SERVE_CONTAINER-$port"
    [ "$(_serve_container_state "$name")" = "running" ] || return 1
  fi
  local live; live="$(serve_models "$port")"
  [ -n "$live" ] && [[ " $live " == *" $served "* ]]
}

serve_receipt() {
  local out="$1"
  if [ "$SERVE_MODE" = "docker" ]; then
    python3 - "$out" "$SERVE_MODE" "$SERVE_IMAGE" "${SERVE_IMAGE_ID:-}" "$SERVE_CONTAINER" "${SERVE_CACHE_VOLUME:-}" "$SERVE_TRUST_REMOTE_CODE" <<'PY'
import json, subprocess, sys
out, mode, image, image_id, container, volume, trc = sys.argv[1:]
insp = json.loads(subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True).stdout or "[{}]")[0]
json.dump({"serve_mode": mode, "image": image, "image_id": image_id, "image_created": insp.get("Created"),
           "repo_digests": insp.get("RepoDigests"), "container": container, "cache_volume": volume,
           "trust_remote_code": trc, "secure_launch": True, "api_auth": True}, open(out, "w"), indent=2)
PY
  else
    python3 - "$out" "$SERVE_MODE" "${SGLANG_DIR:-}" "${ENV_NAME:-}" "$(git -C "${SGLANG_DIR:-.}" rev-parse --short HEAD 2>/dev/null)" <<'PY'
import json, sys
out, mode, tree, env, commit = sys.argv[1:]
json.dump({"serve_mode": mode, "sglang_dir": tree, "env_name": env, "sglang_commit": commit,
           "secure_launch": False, "api_auth": False}, open(out, "w"), indent=2)
PY
  fi
}
