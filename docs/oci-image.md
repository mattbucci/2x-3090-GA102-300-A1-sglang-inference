# OCI image

`Dockerfile` builds the CUDA/v0.5.18 stack without a GPU — every CUDA component is a pinned pip wheel and the driver is injected at `docker run` by the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/). Base images (digest-pinned), the SGLang tag+commit, and the Miniforge installer checksum are pinned; Python/Conda transitive artifacts and live apt repositories are not fully hash-locked, so this is version-constrained rather than bit-reproducible. Adapted from the [R9700 sister repo's](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference) ROCm image — same two-stage shape, secure launcher, and CI promote-by-digest flow — minus their Rust toolchain (patch 037 drops both upstream Rust ext-modules) and minus their sglang API-hardening patch set (see the caveat below). GitHub Actions verifies PR builds and, on main-branch pushes, promotes the exact inspected candidate digest to a full-commit `sha-*` tag at `ghcr.io/<owner>/sglang-cuda-3090`; pin deployments by digest because registry tags remain mutable. **CI activation pending:** the workflow is staged at `.github/workflows-staged/build-image.yaml` because this box's PATs lack the `workflow` scope — `git mv .github/workflows-staged/build-image.yaml .github/workflows/` and push from a workflow-scoped credential to arm it.

```bash
# Local build + offline checks (no GPU needed):
DOCKER_BUILDKIT=1 docker build -t sglang-cuda-3090:local .
python tests/test_secure_launch.py

# Serve a preset (both 3090s, TP=2 preset defaults):
docker run --rm --gpus all \
  -p 127.0.0.1:8000:23334 \
  --cap-drop=ALL --security-opt=no-new-privileges:true \
  --pids-limit 4096 --shm-size 16g \
  -e SGLANG_API_KEY_FILE=/run/secrets/sglang-api-key \
  -e SGLANG_ADMIN_API_KEY_FILE=/run/secrets/sglang-admin-api-key \
  --mount type=bind,src="$api_secret",dst=/run/secrets/sglang-api-key,readonly \
  --mount type=bind,src="$admin_secret",dst=/run/secrets/sglang-admin-api-key,readonly \
  --mount type=bind,src=$HOME/AI/models,dst=/models,readonly \
  sglang-cuda-3090:local \
  scripts/launch.sh qwen36
```

The image runs as unprivileged UID 10001 with `SGLANG_SECURE_LAUNCH=1`: `scripts/launch.sh` routes through `docker/secure-launch.py`, which takes both API keys from files (never argv), refuses protected server options (gRPC/multi-node/LoRA/disaggregation/remote-loader/debug listeners), forces NCCL/GLOO onto loopback, and disables pickle IPC. `--trust-remote-code` and `--enable-metrics` are env-governed in this mode (`-e SGLANG_TRUST_REMOTE_CODE=1`, default 0 — some multimodal presets need it; enable only for reviewed, immutable checkpoints) and the request queue is bounded (`SGLANG_MAX_QUEUED_REQUESTS`, default 32). Two deltas vs the R9700 image to keep in mind: (1) **we do not carry their sglang API-hardening patches** (credential-scrubbed `/get_server_info`, delegated media-source policy, `SGLANG_ALLOW_*_MEDIA` gates) — so their guidance applies doubly here: keep the container on a private network behind an authenticating TLS proxy that denies management routes, and never publish it to an untrusted network; (2) the `developer`-role and list-content chat-template fixes patch **model files**, not sglang — run `scripts/eval/patch_chat_templates_developer_role.py` / `patch_chat_templates_list_content.py --scan` on the host against your models dir before mounting. Bare metal is unaffected by any of this: `SGLANG_SECURE_LAUNCH` defaults to 0 and `launch.sh` argv is byte-identical to the pre-image behavior (verified per-preset via the `DRY_RUN=1` hook).
