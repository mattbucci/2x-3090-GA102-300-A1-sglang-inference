#!/bin/bash
# Disk hygiene for the 3090 box. The two disks fill from two sources:
#   /data (nvme1)  -> /data/models: calibration BF16 bases + AWQ/CT/GPTQ build
#                     intermediates accumulate (1 shipped model leaves ~10 sibling
#                     experiments + a 50-60 GB base).
#   /  (nvme0)     -> /var/lib/docker: SWE-bench per-instance rollout images
#                     accumulate (one per instance, ~5-6 GB, never auto-removed).
#
# SAFETY RULE (set by the user): only ever auto-delete what HuggingFace can serve
# again. Public original BF16 bases (Qwen/google/nvidia repos) are re-downloadable
# -> GC-able. Our in-house REAP/REAM/AWQ/CT/GPTQ outputs are NOT on HF (or only the
# single shipped mattbucci/ copy is) -> NEVER auto-deleted here; they're listed for
# a human to decide. Served models (launch.sh targets) are always kept.
#
# Usage:
#   scripts/maint/disk_hygiene.sh report          # default: show usage + classify
#   scripts/maint/disk_hygiene.sh gc-docker        # prune swebench-rollout images
#   scripts/maint/disk_hygiene.sh gc-docker --all  # prune ALL unused images
#   scripts/maint/disk_hygiene.sh gc-bases         # DRY-RUN: list re-downloadable bases
#   scripts/maint/disk_hygiene.sh gc-bases --apply # delete HF-verified public bases
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-/data/models}"
HF_TOKEN="$(cat ~/.secrets/hf_token 2>/dev/null || true)"
CMD="${1:-report}"; FLAG="${2:-}"

hf_exists() {  # $1 = repo id -> 0 if the model exists on HF
  [ "$(curl -sL -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" \
       "https://huggingface.co/api/models/$1" 2>/dev/null)" = "200" ]
}

# Map a local *-BF16 dir to its candidate upstream repo. Returns "" if the dir
# name carries an in-house marker (never a public base).
candidate_repo() {
  local d="$1"
  case "$d" in
    *REAP*|*REAM*|*AWQ*|*CT*|*GPTQ*|*recal*|*calibrated*|*balanced*|*native*|*AutoRound*|*RTN*|*Marlin*|*Int4*|*int4*) echo ""; return;;
  esac
  case "$d" in
    Qwen3-VL-32B-Instruct-BF16) echo "Qwen/Qwen3-VL-32B-Instruct";;
    gemma-4-31B-it-BF16)        echo "google/gemma-4-31B-it";;
    gemma-4-26B-A4B-it-BF16)    echo "google/gemma-4-26B-A4B-it";;
    Qwen3-Coder-30B-A3B-BF16)   echo "Qwen/Qwen3-Coder-30B-A3B-Instruct";;
    Qwen3.6-27B-BF16)           echo "Qwen/Qwen3.6-27B";;
    Qwen3.5-35B-A3B-BF16)       echo "Qwen/Qwen3.5-35B-A3B";;
    Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) echo "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning";;
    # Unmapped *-BF16: report only (human verifies the repo id before GC).
    *) echo "";;
  esac
}

report() {
  echo "== filesystems =="; df -h / /data 2>/dev/null | grep -vE "^Filesystem" | awk '{printf "  %-8s %5s used, %5s free (%s)\n",$NF,$3,$4,$5}'
  echo "== docker (/var/lib/docker on /) =="
  sudo docker system df 2>/dev/null | awk 'NR<=2'
  echo "  swebench-rollout images: $(sudo docker images --filter=reference='swebench-rollout/*' -q 2>/dev/null | sort -u | wc -l)"
  echo "== /data/models top 15 =="; du -sh "$MODELS_DIR"/* 2>/dev/null | sort -rh | head -15
}

gc_docker() {
  local before; before=$(df -h / | awk 'NR==2{print $4}')
  if [ "$FLAG" = "--all" ]; then
    echo "pruning ALL unused docker images..."; sudo docker image prune -a -f
  else
    echo "pruning swebench-rollout/* images..."
    sudo docker images --filter=reference='swebench-rollout/*' -q 2>/dev/null | sort -u | xargs -r sudo docker rmi -f >/dev/null 2>&1
  fi
  echo "root free: $before -> $(df -h / | awk 'NR==2{print $4}')"
}

gc_bases() {
  local apply=0; [ "$FLAG" = "--apply" ] && apply=1
  echo "scanning $MODELS_DIR/*-BF16 (rule: delete only HF-re-downloadable public originals)"
  for path in "$MODELS_DIR"/*-BF16; do
    [ -d "$path" ] || continue
    local d; d=$(basename "$path"); local sz; sz=$(du -sh "$path" 2>/dev/null | cut -f1)
    local repo; repo=$(candidate_repo "$d")
    if [ -z "$repo" ]; then echo "  KEEP (in-house/unmapped) $sz  $d"; continue; fi
    if hf_exists "$repo"; then
      if [ "$apply" = 1 ]; then rm -rf "$path"; echo "  DELETED $sz  $d  <- $repo"
      else echo "  RE-DOWNLOADABLE $sz  $d  <- $repo  (dry-run; --apply to delete)"; fi
    else
      echo "  KEEP (HF 404, verify repo) $sz  $d  <- $repo"
    fi
  done
}

case "$CMD" in
  report)    report;;
  gc-docker) gc_docker;;
  gc-bases)  gc_bases;;
  *) echo "usage: $0 {report|gc-docker [--all]|gc-bases [--apply]}"; exit 1;;
esac
