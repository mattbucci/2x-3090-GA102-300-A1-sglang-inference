#!/usr/bin/env python3
"""Docker-backed rollout: a coding-agent scaffold runs INSIDE a swebench
instance container. Same env as the official scoring harness (Python
3.6/3.8/3.9/..., conda testbed activated, all system deps installed) —
the model can run pytest mid-iteration against the actual environment
its fix will be graded in.

Per-instance flow:
  1. Pull the upstream eval image  swebench/sweb.eval.x86_64.<normalized>:latest
     (instance_id with __ -> _1776_ per the swebench tag convention).
  2. Build a rollout image  swebench-rollout/<instance_id>:latest  by adding
     Node + opencode + little-coder + ripgrep on top of the eval image
     (Dockerfile.rollout). claw-code lives in Dockerfile.rollout-claw
     because of its Rust build + Anthropic-proxy requirement.
  3. Run a container with --network=host so the scaffold reaches the host
     SGLang server at http://127.0.0.1:23334. Working directory is /testbed
     (already cloned + conda-activated by the upstream eval image).
  4. Exec the scaffold against the problem statement; capture the resulting
     `git diff` as the prediction patch.
  5. Append to predictions.jsonl in the v1 schema + a `rollout_scaffold`
     field so downstream tools can group by scaffold.

Per-(model, scaffold) results are isolated to their own output dir so
predictions, logs, and scores never collide across scaffolds:

    evals/swebench/runs/<preset>-<scaffold>-v2/
        predictions.jsonl                # rollout output
        predictions/<inst>.diff
        logs/<inst>.log
        scores-docker.jsonl              # filled by score_docker.py
        meta.json                        # scaffold + model + run metadata

Usage:
    python evals/swebench/docker_rollout.py \\
        --model sglang/coder-30b-eval \\
        --scaffold opencode \\
        --out evals/swebench/runs/coder-30b-opencode-v2 \\
        --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DOCKERFILE = THIS_DIR / "docker" / "Dockerfile.rollout"
DOCKER_CTX = THIS_DIR / "docker"


# claw-code DEPRECATED 2026-08-30 (unmaintained; kept for reproducing
# historical cells — R9700 retired it the same day). opencode-dcp = opencode
# with the Dynamic Context Pruning plugin (isolated HOME, see Dockerfile).
SUPPORTED_SCAFFOLDS = ("opencode", "opencode-dcp", "little-coder", "little-coder-rtk", "claw-code", "prime", "dcode")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Scaffold-side model id. opencode/little-coder format: "
                        "<provider>/<served-name>. Examples: sglang/coder-30b-eval, "
                        "openai/coder-30b-eval. Provider is opaque to the rollout — "
                        "just controls the scaffold's request routing.")
    p.add_argument("--scaffold", default="opencode", choices=SUPPORTED_SCAFFOLDS,
                   help="Which agent scaffold to invoke inside the container.")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                   help="HF dataset id (Lite=300, Verified=500)")
    p.add_argument("--split", default="test")
    p.add_argument("--instances", type=int, default=0,
                   help="Limit to first N instances (0 = all)")
    p.add_argument("--instance-ids", nargs="*", default=None,
                   help="Specific instance IDs to run (overrides --instances)")
    p.add_argument("--out", required=True,
                   help="Output dir for predictions + logs (per-scaffold convention: "
                        "evals/swebench/runs/<preset>-<scaffold>-v2/)")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-instance scaffold timeout (seconds). v2 opencode "
                        "distribution: p50=121s, p90=210s, p99=1014s, max=1328s — "
                        "1800s catches all but pathological cases. little-coder + "
                        "claw-code default to the same ceiling pending real data.")
    p.add_argument("--server-url", default="http://127.0.0.1:23334",
                   help="SGLang server base URL (used for preflight)")
    p.add_argument("--served-name", default=None,
                   help="Served model name (defaults to model id after slash)")
    p.add_argument("--context-window", type=int, default=0,
                   help="Token budget every scaffold is told the model has. 0 (default) "
                        "= read max_model_len from the server's /v1/models so the "
                        "scaffold budget always equals the served KV window. Found "
                        "2026-09-11: little-coder's pi ran every preset at a 32,768 "
                        "fallback (unknown model id -> first packaged models.json "
                        "entry), prime-agent defaulted to 128K, dcode to a 170K "
                        "summarization trigger, and opencode.json carried 32K for "
                        "several presets. Budget is now injected per run (see "
                        "build_scaffold_invocation) and recorded in meta.json.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip instances that already have a prediction")
    p.add_argument("--max-empty-streak", type=int, default=10,
                   help="Abort after this many consecutive empty diffs")
    p.add_argument("--keep-containers", action="store_true",
                   help="Don't `docker rm` after each instance (debug)")
    p.add_argument("--no-pull", action="store_true",
                   help="Skip `docker pull` of the upstream eval image (use only if "
                        "already cached locally; otherwise the build will fail)")
    p.add_argument("--rebuild-image", action="store_true",
                   help="Force `docker build` even when the rollout image already "
                        "exists. Use after Dockerfile.rollout changes (e.g. adding "
                        "a new scaffold) so the layered binaries are present.")
    return p.parse_args()


# Activate the SWE-bench `testbed` conda env EXPLICITLY inside the container
# script. `bash -lc` only reaches `conda activate testbed` through
# /root/.bashrc; the lanes that override HOME for config isolation
# (opencode-dcp -> /opt/dcp-home, little-coder-rtk -> /opt/rtk-home) skipped it
# and ran the scaffold on the miniconda BASE interpreter while the prompt
# promised the repo env. qwen38 receipts (2026-09-07): "No module named
# <repo>" in 34% of DCP sessions vs 10% opencode control; env-hunting in 85%
# of RTK ledgers; 5 timeouts in the first 40 RTK instances vs 0 control.
# Idempotent for HOME=/root lanes (same env the login shell already gave).
ACTIVATE_TESTBED = (
    "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then\n"
    "  source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed\n"
    "fi\n"
)


# Uniform per-request output budget for every scaffold. Thinking models emit
# long reasoning traces: opencode's packaged `limit.output: 8192` truncated
# 38 of 293 qwen38 opencode sessions, 35 of which ended as empty patches (69%
# of that cell's empties; 62% on the DCP lane), pi's 16384 hit the cap 29x
# in the first 250 little-coder sessions. Must EQUAL each scaffold's compaction
# reserve: SGLang 400s any request whose prompt + max_tokens exceeds the served
# window (presets deliberately don't pass --allow-auto-truncate, which would
# also silently truncate over-long prompts), so the scaffold has to compact
# before the prompt reaches window - OUTPUT_BUDGET.
OUTPUT_BUDGET = 32768
# What the wire actually carries: pi-ai 0.68 (little-coder 1.1.0) and
# prime-agent clamp `Math.min(model.maxTokens, 32000)` in simple-options.js;
# the current @earendil-works pi (rtk lane) and opencode send OUTPUT_BUDGET
# as-is. scaffold_request_audit.py checks caps against this floor.
OUTPUT_CAP_FLOOR = 32000

# Thinking policy: every scaffold runs the model at its MAXIMUM thinking
# tier. No scaffold may send `reasoning_effort` (pi's default "medium"
# halved qwen38's budget; pi also clamps xhigh->high, which the Qwen3.8
# template rejects) or `enable_thinking: false`. Sending nothing lets the
# served chat template's default apply — the preset owns that default
# (Qwen3.8 -> xhigh, Qwen3.5/3.6 -> on; gemma4 presets pass
# --default-chat-template-kwargs). scaffold_request_audit.py asserts this.


def _pi_settings_snippet(agent_dir: str) -> str:
    """Shell lines that merge `compaction.reserveTokens = OUTPUT_BUDGET` into
    a pi-family global settings.json (pi-coding-agent / prime-agent both
    default 16384: compaction fires at contextWindow - reserveTokens, so the
    reserve must cover the request's max_completion_tokens or the server
    rejects the turn). `agent_dir` is a shell expression ($HOME/.pi/agent,
    /root/.prime/agent)."""
    js = (
        'const fs=require("fs");const d=process.argv[1];const p=d+"/settings.json";'
        'fs.mkdirSync(d,{recursive:true});'
        'let s={};try{s=JSON.parse(fs.readFileSync(p,"utf8"))}catch(e){}'
        f's.compaction=Object.assign({{}},s.compaction||{{}},{{reserveTokens:{OUTPUT_BUDGET}}});'
        'fs.writeFileSync(p,JSON.stringify(s,null,2));'
        'console.error("context budget: "+p+" compaction="+JSON.stringify(s.compaction));'
    )
    return f"node -e '{js}' \"{agent_dir}\"\n"


def _opencode_budget_snippet(served_name: str, context_window: int) -> str:
    """Shell lines that pin `provider.sglang.models[<served>].limit.context`
    in the lane's ~/.config/opencode/opencode.json to the served window
    (adding the entry if the preset is missing — the old "new preset needs
    an opencode.json entry + image nuke" landmine) and `limit.output` to
    OUTPUT_BUDGET (opencode sends max_tokens = min(limit.output,
    OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX) and auto-compacts at
    limit.context - that). HOME-relative, so the DCP lane's /opt/dcp-home
    copy is the one edited there."""
    js = (
        'const fs=require("fs");const p=process.env.HOME+"/.config/opencode/opencode.json";'
        'const d=JSON.parse(fs.readFileSync(p,"utf8"));const m=d.provider.sglang.models;'
        f'const id={json.dumps(served_name)};const e=m[id]||(m[id]={{name:id,tool_call:true}});'
        f'e.limit=Object.assign({{}},e.limit||{{}},{{context:{int(context_window)},output:{OUTPUT_BUDGET}}});'
        'fs.writeFileSync(p,JSON.stringify(d,null,2));'
        'console.error("context budget: opencode.json "+id+" limit="+JSON.stringify(e.limit));'
    )
    return f"node -e '{js}'\n"


def _little_coder_budget_snippet(pkg_root: str, served_name: str, context_window: int) -> str:
    """Shell lines that write a per-run little-coder models file declaring the
    served model under the llamacpp provider (copying the packaged provider's
    api/baseUrl/apiKey) and export LITTLE_CODER_MODELS_FILE so both little-coder
    1.1.0 and 1.19.0 load it (provider-level merge over the packaged file).
    Without it pi's model resolver clones the first packaged entry: 32K.
    `compat.supportsReasoningEffort:false` keeps pi from sending its default
    `reasoning_effort: medium` (see the thinking policy above); maxTokens and
    the settings.json compaction reserve are both OUTPUT_BUDGET."""
    js = (
        f'const fs=require("fs");const pkg=JSON.parse(fs.readFileSync({json.dumps(pkg_root + "/models.json")},"utf8"));'
        f'const prov=pkg.providers.llamacpp;const id={json.dumps(served_name)};'
        f'prov.models=[{{id:id,name:id+" (SGLang, served window)",reasoning:true,input:["text"],'
        f'contextWindow:{int(context_window)},maxTokens:{OUTPUT_BUDGET},cost:{{input:0,output:0,cacheRead:0,cacheWrite:0}},'
        'compat:{supportsReasoningEffort:false}}];'
        'fs.writeFileSync("/tmp/sweb-little-coder-models.json",JSON.stringify({providers:{llamacpp:prov}},null,2));'
        'console.error("context budget: little-coder models.json "+id+" contextWindow="+prov.models[0].contextWindow);'
    )
    return (f"node -e '{js}'\nexport LITTLE_CODER_MODELS_FILE=/tmp/sweb-little-coder-models.json\n"
            + _little_coder_profile_snippet(pkg_root, served_name, context_window)
            + _pi_settings_snippet("$HOME/.pi/agent"))


# little-coder per-model profile pinned for the served model (R9700 finding,
# their commit 96a61f5, 2026-09-13; verified in both of our prefixes). The
# package's benchmark-profiles extension resolves `little_coder.model_profiles`
# from the PACKAGE's own .pi/settings.json (exact key, then prefix, then
# `default_model_profile`); an unknown served model gets the default profile:
# 1.1.0 = thinking_budget 2048 / context_limit 32768 / temperature 0.3,
# 1.19.0 = thinking_budget 4096 / temperature 0.3. The thinking-budget
# extension counts thinking_delta chars/3.5 and on breach ABORTS the turn,
# flips pi's thinking level to "off" (which only drops `reasoning_effort` —
# we never send it, so the template keeps thinking at its default) and queues
# "[thinking budget exceeded] Please commit to an implementation now" — at
# xhigh that is an abort->nudge loop until the rollout timeout (R9700: 3258
# iterations in 90 s on a ~6K-token thinking stream). The profile beats
# LITTLE_CODER_THINKING_BUDGET (profile || env || default), so pinning the
# key is the only lever. No `temperature`: the extension then injects
# nothing and SGLang applies the checkpoint's generation_config, the same
# sampling every other lane gets (lanes before 2026-09-13 ran at T=0.3).
LC_MODEL_PROFILE = {
    "max_tokens": OUTPUT_BUDGET,   # informational; the wire cap is models.json maxTokens
    "thinking_budget": 1000000,    # never trips the abort (server-bound instead)
    "skill_token_budget": 300,     # unchanged package defaults from here down
    "knowledge_token_budget": 200,
    "system_prompt_budget": 0,
    "max_retries": 1,
}


def _little_coder_profile_snippet(pkg_root: str, served_name: str, context_window: int) -> str:
    """Shell lines that pin `little_coder.model_profiles["llamacpp/<served>"]`
    in the package's .pi/settings.json (see LC_MODEL_PROFILE). The image is
    rebuilt per instance, so this runs on every rollout; the file lives inside
    the npm package (an upgrade drops it)."""
    prof = dict(LC_MODEL_PROFILE, context_limit=int(context_window))
    js = (
        f'const fs=require("fs");const p={json.dumps(pkg_root + "/.pi/settings.json")};'
        'const d=JSON.parse(fs.readFileSync(p,"utf8"));const lc=d.little_coder||(d.little_coder={});'
        f'const mp=lc.model_profiles||(lc.model_profiles={{}});const k="llamacpp/"+{json.dumps(served_name)};'
        f'mp[k]={json.dumps(prof, separators=(",", ":"))};'
        'fs.writeFileSync(p,JSON.stringify(d,null,2)+"\\n");'
        'console.error("context budget: little-coder model profile "+k+" "+JSON.stringify(mp[k]));'
    )
    return f"node -e '{js}'\n"


def build_scaffold_invocation(scaffold: str, model: str, served_name: str,
                              timeout: int = 1800, context_window: int = 0) -> tuple[list[str], str]:
    """Return (docker_run_extra_envs, inner_shell_command) for the given
    scaffold. The inner command runs opencode-equivalent against $PROMPT
    in /testbed and emits `=== DIFF ===\\n<git diff>` to stdout for the
    parent process to extract.

    All scaffolds share the same diff-capture protocol so the parent
    `_extract_diff_from_stdout` works uniformly, and every scaffold is told
    the same `context_window` (the served KV window) through its own config
    surface — opencode.json limit.context, little-coder models file,
    prime models.json contextWindow, dcode --profile-override.
    """
    if context_window <= 0:
        raise ValueError("context_window must be the served window (>0); see --context-window")
    if scaffold == "opencode":
        # opencode reads ~/.config/opencode/opencode.json which the Dockerfile
        # provisions. The `sglang/<served-name>` model id wires there.
        envs = ["--env", f"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX={OUTPUT_BUDGET}"]
        inner = (
            f"set -e\n"
            f"git config --global user.email eval@local\n"
            f"git config --global user.name eval\n"
            f"git config --global --add safe.directory /testbed\n"
            + _opencode_budget_snippet(served_name, context_window) +
            f"opencode run --dir /testbed --model {model} "
            f"  --format json --dangerously-skip-permissions \"$PROMPT\" || true\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"timeout 120 opencode run --dir /testbed --model {model} "
            f"  --format json --dangerously-skip-permissions \"$CLEANUP_PROMPT\" || true\n"
            f"echo === DIFF ===\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"git -C /testbed add -A\n"
            f"git -C /testbed diff --cached\n"
        )
        return envs, inner

    if scaffold == "little-coder":
        # little-coder wraps pi-ai, which reads provider baseUrls from a
        # packaged models.json — not from OPENAI_BASE_URL. Dockerfile.rollout
        # repoints the "llamacpp" provider's baseUrl at our SGLang endpoint,
        # so we route through llamacpp/<served-name>. The served id is
        # declared in a per-run models file (LITTLE_CODER_MODELS_FILE) with the
        # served context window: the old "Using custom model id" fallback was
        # NOT benign — pi cloned the first packaged entry and ran every preset
        # at contextWindow 32768 (compaction loops / aborted sessions; see
        # benchmarks/quality/rtk-lane-close-qwen38-2026-09-11.md).
        oc_model = f"llamacpp/{served_name}"
        envs = [
            "--env", "LLAMACPP_API_KEY=noop",
        ]
        inner = (
            f"set -e\n"
            f"git config --global user.email eval@local\n"
            f"git config --global user.name eval\n"
            f"git config --global --add safe.directory /testbed\n"
            + _little_coder_budget_snippet("/opt/node/lib/node_modules/little-coder", served_name, context_window) +
            f"cd /testbed\n"
            f"little-coder --model {oc_model} \"$PROMPT\" || true\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"timeout 120 little-coder --model {oc_model} \"$CLEANUP_PROMPT\" || true\n"
            f"echo === DIFF ===\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"git -C /testbed add -A\n"
            f"git -C /testbed diff --cached\n"
        )
        return envs, inner

    if scaffold == "little-coder-rtk":
        # little-coder with the RTK bash-output compressor loaded EXPLICITLY
        # via pi's -e flag: headless little-coder runs skip pi's extension
        # auto-discovery (proven with a marker extension), and the session
        # jsonl records the PRE-mutation command — engagement was verified
        # with an rtk-shim log showing `rtk rewrite git status` -> executed
        # `rtk git status` (2026-08-31).
        # Uses the lane's OWN little-coder 1.19.0 prefix (/opt/lc-rtk): rtk's
        # extension needs the current @earendil-works pi; the control lane's
        # 1.1.0 bundle (@mariozechner pi) ignores it — verified, commands
        # stayed unrewritten. HOME override wins (scaffold envs append after
        # --env HOME=/root; docker takes the last occurrence).
        oc_model = f"llamacpp/{served_name}"
        envs = [
            "--env", "HOME=/opt/rtk-home",
            "--env", "LLAMACPP_API_KEY=noop",
            "--env", "RTK_TELEMETRY_DISABLED=1",
        ]
        inner = (
            f"set -e\n"
            f"git config --global user.email eval@local\n"
            f"git config --global user.name eval\n"
            f"git config --global --add safe.directory /testbed\n"
            + _little_coder_budget_snippet("/opt/lc-rtk/node_modules/little-coder", served_name, context_window) +
            f"cd /testbed\n"
            f"/opt/lc-rtk/node_modules/.bin/little-coder -e /opt/rtk-home/.pi/agent/extensions/rtk.ts --model {oc_model} \"$PROMPT\" || true\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"timeout 120 /opt/lc-rtk/node_modules/.bin/little-coder -e /opt/rtk-home/.pi/agent/extensions/rtk.ts --model {oc_model} \"$CLEANUP_PROMPT\" || true\n"
            f"echo === DIFF ===\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"git -C /testbed add -A\n"
            f"git -C /testbed diff --cached\n"
        )
        return envs, inner

    if scaffold == "claw-code":
        # claw natively supports OpenAI-compat: OPENAI_BASE_URL +
        # OPENAI_API_KEY, model id "openai/<served>". The openai/ prefix
        # wins over the ambient credential sniffer, so DashScope's
        # qwen-/qwen prefix routing won't intercept (claw USAGE.md
        # "Provider matrix" + PR 3001 reasoning_content support). The
        # rollout image must include the pre-built claw binary at
        # /usr/local/bin/claw — Dockerfile.rollout COPYs it.
        oc_model = f"openai/{served_name}"
        envs = [
            "--env", "OPENAI_BASE_URL=http://127.0.0.1:23334/v1",
            "--env", "OPENAI_API_KEY=noop",
        ]
        inner = (
            f"set -e\n"
            f"git config --global user.email eval@local\n"
            f"git config --global user.name eval\n"
            f"git config --global --add safe.directory /testbed\n"
            f"cd /testbed\n"
            f"/usr/local/bin/claw --model {oc_model} prompt \"$PROMPT\" || true\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"timeout 120 /usr/local/bin/claw --model {oc_model} prompt \"$CLEANUP_PROMPT\" || true\n"
            f"echo === DIFF ===\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"git -C /testbed add -A\n"
            f"git -C /testbed diff --cached\n"
        )
        return envs, inner

    if scaffold == "opencode-dcp":
        # Same invocation as opencode, but HOME points at the DCP-enabled
        # config home baked by Dockerfile.rollout (plugin + dcp.jsonc). The
        # scaffold envs are appended AFTER --env HOME=/root in the docker cmd,
        # and docker takes the last occurrence, so this override wins.
        envs = ["--env", "HOME=/opt/dcp-home",
                "--env", f"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX={OUTPUT_BUDGET}"]
        inner = (
            f"set -e\n"
            f"git config --global user.email eval@local\n"
            f"git config --global user.name eval\n"
            f"git config --global --add safe.directory /testbed\n"
            + _opencode_budget_snippet(served_name, context_window) +
            f"opencode run --dir /testbed --model {model} "
            f"  --format json --dangerously-skip-permissions \"$PROMPT\" || true\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"timeout 120 opencode run --dir /testbed --model {model} "
            f"  --format json --dangerously-skip-permissions \"$CLEANUP_PROMPT\" || true\n"
            f"echo === DIFF ===\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"git -C /testbed add -A\n"
            f"git -C /testbed diff --cached\n"
        )
        return envs, inner

    if scaffold == "prime":
        # prime-agent reads custom providers from ~/.prime/agent/models.json
        # (no config-dir override as of 0.8.1) — written HERE with the actual
        # served name so new presets need no static enumeration. compat flags
        # follow the SGLang guidance (ported from R9700's host-side runner,
        # verified by them on prime-agent 0.8.1). DO_NOT_TRACK: prime sends
        # pseudonymous usage metrics by default; eval rollouts don't phone home.
        provider_json = json.dumps({
            "providers": {"sglang": {
                "name": "SGLang local",
                "baseUrl": "http://127.0.0.1:23334/v1",
                "api": "openai-completions",
                "apiKey": "noop",
                "compat": {"supportsDeveloperRole": False,
                           "supportsReasoningEffort": False},
                # explicit budget: prime-agent's custom-provider default is
                # contextWindow 128000 / maxTokens 16384 (model-registry.js)
                "models": [{"id": served_name,
                            "contextWindow": int(context_window),
                            "maxTokens": OUTPUT_BUDGET}],
            }}}, indent=2)
        envs = ["--env", "DO_NOT_TRACK=1"]
        inner = (
            "set -e\n"
            "git config --global user.email eval@local\n"
            "git config --global user.name eval\n"
            "git config --global --add safe.directory /testbed\n"
            "mkdir -p /root/.prime/agent\n"
            "cat > /root/.prime/agent/models.json <<'PRIMEJSON'\n"
            f"{provider_json}\n"
            "PRIMEJSON\n"
            + _pi_settings_snippet("/root/.prime/agent") +
            "cd /testbed\n"
            f"prime-agent --model sglang/{served_name} -p \"$PROMPT\" || true\n"
            "rm -rf /testbed/.prime /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"timeout 120 prime-agent --model sglang/{served_name} -p \"$CLEANUP_PROMPT\" || true\n"
            "echo === DIFF ===\n"
            "rm -rf /testbed/.prime /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            "git -C /testbed add -A\n"
            "git -C /testbed diff --cached\n"
        )
        return envs, inner

    if scaffold == "dcode":
        # deepagents-code headless: -n runs one task and exits (-q clean
        # output); tools auto-run headless. Model routing is plain openai-SDK
        # env (OPENAI_BASE_URL + `openai:<served>`; SGLang ignores the id).
        # The inner --timeout is derived from the outer per-instance timeout
        # (outer minus 100 s) so dcode always exits on its own before the
        # outer SIGKILL — a fixed inner value bit the 900 s smoke runs
        # (rc=124 mid-session -> empty diff). Ported from R9700, 0.1.65.
        envs = [
            "--env", "OPENAI_BASE_URL=http://127.0.0.1:23334/v1",
            "--env", "OPENAI_API_KEY=noop",
        ]
        inner = (
            "set -e\n"
            "git config --global user.email eval@local\n"
            "git config --global user.name eval\n"
            "git config --global --add safe.directory /testbed\n"
            "cd /testbed\n"
            # --profile-override: langchain has no profile for a custom
            # openai:<id>, so deepagents falls back to a fixed 170K-token
            # summarization trigger; with max_input_tokens set it uses
            # fraction 0.85 of the served window.
            f"dcode -M openai:{served_name} --profile-override '{{\"max_input_tokens\": {int(context_window)}}}' -n \"$PROMPT\" -q --max-turns 60 -S all --allow-fs-tools all --timeout {max(60, timeout - 100)} || true\n"
            "rm -rf /testbed/.deepagents /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"timeout 120 dcode -M openai:{served_name} --profile-override '{{\"max_input_tokens\": {int(context_window)}}}' -n \"$CLEANUP_PROMPT\" -q --max-turns 8 -S all --allow-fs-tools all --timeout 100 || true\n"
            "echo === DIFF ===\n"
            "rm -rf /testbed/.deepagents /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            "git -C /testbed add -A\n"
            "git -C /testbed diff --cached\n"
        )
        return envs, inner

    raise ValueError(f"unknown scaffold: {scaffold}")


# --- helpers ---------------------------------------------------------------

def sh(*args, check=True, capture=False, cwd=None, env=None, timeout=None):
    """subprocess.run wrapper. capture=True returns (rc, stdout, stderr)."""
    kwargs = {
        "cwd": cwd,
        "env": env,
        "timeout": timeout,
    }
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    proc = subprocess.run(list(args), check=False, **kwargs)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args,
                                            output=proc.stdout if capture else None,
                                            stderr=proc.stderr if capture else None)
    if capture:
        return proc.returncode, proc.stdout, proc.stderr
    return proc.returncode


def _swebench_image_tag(instance_id: str) -> str:
    """Upstream eval image tag for a SWE-bench instance.

    SWE-bench normalizes `<org>__<repo>-<n>` to `<org>_1776_<repo>-<n>` for
    Docker tag compatibility (image tags can't contain `__` consistently).
    """
    normalized = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{normalized}:latest"


def rollout_image_tag(instance_id: str) -> str:
    """Local rollout image tag (eval image + Node + opencode + ripgrep)."""
    return f"swebench-rollout/{instance_id}:latest"


def ensure_rollout_image(instance_id: str, *, no_pull: bool = False,
                         rebuild: bool = False) -> str:
    """Build the per-instance rollout image if not already present.

    Idempotent: docker build is a no-op when the tag exists with all layers
    cached, so re-runs after the first instance build are fast. With
    rebuild=True, forces a fresh build even if the tag already resolves —
    use after Dockerfile.rollout changes (e.g. adding a new scaffold).
    """
    rollout_tag = rollout_image_tag(instance_id)
    base_tag = _swebench_image_tag(instance_id)

    if not rebuild:
        rc, out, _ = sh("docker", "image", "inspect", rollout_tag,
                        check=False, capture=True)
        if rc == 0:
            return rollout_tag

    if not no_pull:
        rc_pull = sh("docker", "pull", base_tag, check=False)
        if rc_pull != 0:
            raise RuntimeError(f"docker pull {base_tag} failed (rc={rc_pull})")

    sh("docker", "build",
       "-t", rollout_tag,
       "--build-arg", f"BASE={base_tag}",
       "-f", str(DOCKERFILE),
       str(DOCKER_CTX),
       check=True)
    return rollout_tag


PROMPT_TEMPLATE = """\
You are working on a GitHub issue in this repository.

The repo is already installed in editable mode in the active conda environment
on your PATH (testbed). You can run `pytest` and `python -c "..."` to verify
imports, exercise edge cases, and re-run tests after each edit. Use this:
write a fix, run the relevant tests, observe failures, refine until green.

Read the problem carefully, locate the relevant code, and write the minimal
patch that fixes the bug. Do not modify tests. Do not add new files unless
strictly required. When you're confident the fix is correct AND the tests
exercise it correctly, stop — your final state will be captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""


# Self-clean pass: invoked AFTER the original prompt finishes. Same scaffold,
# same model, fresh session. The model inspects git state and rm's its own
# reproducer/debug helpers so they don't make it into the captured diff. Why
# this matters: SWE-bench's grader marks an instance "error" (not unresolved)
# when pytest collects a model-written helper at /testbed root that errors
# at import time, because then the harness can't find FAIL_TO_PASS test
# results in the malformed log. See evals/swebench/filter_predictions.py
# for the score-time regex fallback that catches what self-clean misses.
CLEANUP_PROMPT = """\
Cleanup step. You just finished fixing a bug in this Python repository. The original task said "Do not add new files unless strictly required" — but during exploration you may have created reproducer / debug / analysis scripts. Clean those up now so only your real fix remains.

Steps:
1. Run `git status` to see what changed.
2. For each NEW file at the repository root (or other non-standard location) that looks like a model-generated helper — names matching reproduce_*, debug_*, analyze_*, test_fix*, simple_test_*, comprehensive_*, check_*, *_bug.py, etc. — delete it with `rm`.
3. Do NOT touch modifications to pre-existing tracked files: those are your real fix.
4. Do NOT delete files inside the official test directory (tests/, test_*/, *_test/).

After cleanup, stop. Don't run pytest, don't edit any code.
"""


def run_in_container(image_tag: str, instance_id: str, prompt: str, model: str,
                     timeout: int, log_path: Path, *, keep: bool = False) -> tuple[int, str, str]:
    """Run opencode inside the rollout container; return (rc, stdout, stderr).

    The container runs in a fresh process group so we can SIGKILL the whole
    docker subtree on timeout. Default removes the container after exit;
    --keep-containers preserves them for debugging.
    """
    container_name = f"swebench-rollout-{instance_id}-{int(time.time())}"
    rm_flag = [] if keep else ["--rm"]

    # `git config --global` ensures the diff-on-exit path doesn't trip on
    # missing user.email/name inside the container.
    inner = f"""set -e
git config --global user.email eval@local
git config --global user.name eval
git config --global --add safe.directory /testbed
opencode run --dir /testbed --model {model} --format json --dangerously-skip-permissions "$PROMPT"
"""

    cmd = [
        "docker", "run",
        *rm_flag,
        "--name", container_name,
        "--network=host",
        "--env", f"PROMPT={prompt}",
        "--env", "HOME=/root",
        "--workdir", "/testbed",
        image_tag,
        "bash", "-lc", ACTIVATE_TESTBED + inner,
    ]

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
        elapsed = time.time() - t0
        log_path.write_text(
            f"# command (PROMPT in env)\n{' '.join(cmd[:-3])} bash -lc <inner>\n"
            f"# elapsed {elapsed:.1f}s\n# returncode {rc}\n"
            f"# stdout\n{stdout}\n# stderr\n{stderr}\n"
        )
        return rc, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # Belt-and-suspenders: ask docker to kill the container too. The
        # docker CLI process getting SIGKILLed doesn't necessarily reap the
        # daemon-managed container.
        sh("docker", "kill", container_name, check=False, capture=True)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        log_path.write_text(
            f"# TIMEOUT after {timeout}s (process group + docker kill)\n"
            f"# stdout\n{stdout}\n# stderr\n{stderr}\n"
        )
        return 124, stdout or "", stderr or ""


def capture_diff(image_tag: str, instance_id: str) -> str:
    """Run a fresh container against the same image to capture a `git diff` is
    not viable — the rollout container is gone (--rm) and held the worktree.
    Instead we capture inside the same `inner` shell script in run_in_container
    by appending `git add -A && git diff --cached`. This function exists only
    to keep the v1 schema symmetry; the actual diff capture happens in
    run_in_container's stdout (parsed below)."""
    raise NotImplementedError("diff capture is inlined into run_in_container; see _extract_diff_from_stdout")


def _extract_diff_from_stdout(stdout: str) -> str:
    """The inner shell appends `=== DIFF ===` then `git diff --cached` output.
    Strip everything before that marker."""
    marker = "=== DIFF ==="
    idx = stdout.rfind(marker)
    if idx == -1:
        return ""
    return stdout[idx + len(marker):].lstrip("\n")


# --- preflight + dataset ---------------------------------------------------

def preflight_canary(server_url: str, served_name: str) -> tuple[bool, str]:
    """Mimic opencode's wire format (assistant turn with prior tool_calls,
    arguments as JSON string per OpenAI spec) to catch chat-template bugs
    before burning hours on rollouts. Same shape as run_rollouts.py."""
    payload = {
        "model": served_name,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "glob",
                                          "arguments": '{"pattern": "**/*.py"}'}}]},
            {"role": "tool", "tool_call_id": "1", "content": "a.py\nb.py"},
            {"role": "user", "content": "continue"},
        ],
        "max_tokens": 30,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            if "choices" in body and body["choices"]:
                content = body["choices"][0]["message"].get("content") or ""
                return True, f"OK ({len(content)}B content)"
            return False, f"unexpected response: {body!r}"
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read())["message"]
        except Exception:
            err = str(e)
        return False, f"{e.code}: {err}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def served_context_window(server_url: str, served: str) -> int | None:
    """max_model_len the server advertises for `served` on /v1/models (SGLang
    reports the launch --context-length there). None if unavailable."""
    try:
        with urllib.request.urlopen(f"{server_url}/v1/models", timeout=30) as r:
            body = json.loads(r.read())
    except Exception:
        return None
    cards = body.get("data") or []
    for c in cards:
        if c.get("id") == served and c.get("max_model_len"):
            return int(c["max_model_len"])
    if len(cards) == 1 and cards[0].get("max_model_len"):
        return int(cards[0]["max_model_len"])
    return None


# pi (little-coder) prints this when the model id is not in the provider's
# models.json and it clones the first entry (32K). With the per-run models
# file below it must never appear; if it does, the context budget was NOT
# applied and the instance ran at 32K.
PI_FALLBACK_MARKER = "not found for provider"


def load_dataset(dataset_id: str, split: str):
    from datasets import load_dataset as _ld
    return _ld(dataset_id, split=split)


# --- main ------------------------------------------------------------------

def main():
    args = parse_args()

    served = args.served_name or args.model.split("/", 1)[-1]
    print(f"Preflight: canary chat completion against {args.server_url} (model={served})...", flush=True)
    ok, info = preflight_canary(args.server_url, served)
    if not ok:
        print(f"  PREFLIGHT FAILED: {info}", flush=True)
        print(f"  refusing to start rollout — fix the server / chat template first", flush=True)
        return 2
    print(f"  preflight {info}", flush=True)

    if args.context_window > 0:
        ctx, ctx_src = args.context_window, "--context-window"
    else:
        ctx, ctx_src = served_context_window(args.server_url, served), "server /v1/models max_model_len"
    if not ctx:
        print("  CONTEXT BUDGET UNKNOWN: /v1/models did not report max_model_len — pass --context-window", flush=True)
        return 2
    print(f"Context budget: {ctx} tokens for every scaffold (source: {ctx_src})", flush=True)

    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    # meta.json declares scaffold + model + run dates so future tooling can
    # group predictions.jsonl files by scaffold without inferring from the
    # parent dirname. Idempotent: appends a new run record on resume.
    meta_path = out / "meta.json"
    meta = {"runs": []}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    meta.setdefault("runs", []).append({
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scaffold": args.scaffold,
        "model": args.model,
        "served_name": args.served_name or args.model.split("/", 1)[-1],
        "dataset": args.dataset,
        "split": args.split,
        "timeout_sec": args.timeout,
        "context_window": ctx,
        "context_window_source": ctx_src,
        # harness policy this run rolled under (see OUTPUT_BUDGET / thinking
        # policy above; scaffold_request_audit.py proves it on the wire)
        "output_budget": OUTPUT_BUDGET,
        "thinking": "max (template default; no reasoning_effort sent)",
    })
    meta["scaffold"] = args.scaffold
    meta["model"] = args.model
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Scaffold: {args.scaffold}   Model: {args.model}   Out: {out}", flush=True)

    print(f"Loading dataset {args.dataset}/{args.split}...", flush=True)
    ds = load_dataset(args.dataset, args.split)
    print(f"  {len(ds)} instances total", flush=True)

    if args.instance_ids:
        ds = [r for r in ds if r["instance_id"] in args.instance_ids]
        print(f"  filtered to {len(ds)} via --instance-ids", flush=True)
    elif args.instances:
        ds = list(ds)[: args.instances]
        print(f"  truncated to first {len(ds)} via --instances", flush=True)

    predictions_path = out / "predictions.jsonl"
    existing = set()
    if args.skip_existing and predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            try:
                existing.add(json.loads(line)["instance_id"])
            except Exception:
                pass
        print(f"  resume: {len(existing)} predictions already on disk", flush=True)

    empty_streak = 0
    with predictions_path.open("a") as fp:
        for i, row in enumerate(ds):
            iid = row["instance_id"]
            if iid in existing:
                print(f"[{i+1}/{len(ds)}] {iid}  SKIP (exists)", flush=True)
                continue

            print(f"[{i+1}/{len(ds)}] {iid}  repo={row['repo']}  base={row['base_commit'][:8]}", flush=True)
            t0 = time.time()
            try:
                image_tag = ensure_rollout_image(
                    iid, no_pull=args.no_pull, rebuild=args.rebuild_image,
                )
                prompt = PROMPT_TEMPLATE.format(
                    problem_statement=row["problem_statement"],
                    hints=row.get("hints_text", "") or "(none)",
                )
                scaffold_envs, inner_with_diff = build_scaffold_invocation(
                    args.scaffold, args.model, served, timeout=args.timeout,
                    context_window=ctx,
                )
                container_name = f"swebench-rollout-{iid}-{int(time.time())}"
                cmd = [
                    "docker", "run",
                    *([] if args.keep_containers else ["--rm"]),
                    "--name", container_name,
                    "--network=host",
                    "--env", f"PROMPT={prompt}",
                    "--env", f"CLEANUP_PROMPT={CLEANUP_PROMPT}",
                    "--env", "HOME=/root",
                    *scaffold_envs,
                    "--workdir", "/testbed",
                    image_tag,
                    "bash", "-lc", ACTIVATE_TESTBED + inner_with_diff,
                ]
                log_path = out / "logs" / f"{iid}.log"
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    errors="replace",
                    start_new_session=True,
                )
                try:
                    stdout, stderr = proc.communicate(timeout=args.timeout)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    sh("docker", "kill", container_name, check=False, capture=True)
                    try:
                        stdout, stderr = proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        stdout, stderr = "", ""
                    rc = 124

                elapsed = round(time.time() - t0, 1)
                log_path.write_text(
                    f"# command (PROMPT in env)\n"
                    f"# elapsed {elapsed}s   rc={rc}\n"
                    f"# stdout\n{stdout}\n# stderr\n{stderr}\n"
                )

                if args.scaffold.startswith("little-coder") and PI_FALLBACK_MARKER in (stderr + stdout):
                    print(f"  CONTEXT-BUDGET TRIPWIRE: pi reported '{PI_FALLBACK_MARKER}' — the per-run "
                          f"models file was not honoured; this instance ran at the 32K fallback", flush=True)

                diff = _extract_diff_from_stdout(stdout)
                (out / "predictions" / f"{iid}.diff").write_text(diff)
                entry = {
                    "instance_id": iid,
                    "model_name_or_path": args.model,
                    "model_patch": diff,
                    "rollout_returncode": rc,
                    "rollout_seconds": elapsed,
                    "rollout_scaffold": args.scaffold,
                }
                fp.write(json.dumps(entry) + "\n")
                fp.flush()

                non_empty = "yes" if diff.strip() else "EMPTY"
                print(f"  done rc={rc} elapsed={elapsed}s diff={non_empty} ({len(diff)}B)", flush=True)

                if diff.strip():
                    empty_streak = 0
                else:
                    empty_streak += 1
                    if empty_streak >= args.max_empty_streak:
                        print(f"\nABORT: {empty_streak} consecutive empty diffs — re-run preflight before resuming.", flush=True)
                        return 3
            except Exception as e:
                import traceback
                print(f"  SKIP (instance crashed): {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                fp.write(json.dumps({"instance_id": iid, "model_name_or_path": args.model,
                                     "model_patch": "", "rollout_returncode": -1,
                                     "rollout_error": f"{type(e).__name__}: {e}",
                                     "rollout_seconds": round(time.time() - t0, 1)}) + "\n")
                fp.flush()
                continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
