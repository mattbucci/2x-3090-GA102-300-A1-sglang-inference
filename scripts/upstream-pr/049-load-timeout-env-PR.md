# Upstream PR draft — env override for UNBALANCED_MODEL_LOADING_TIMEOUT_S (our patch 049)

**Status: PREPARED, NOT OPENED** — user green-light required.
Acceptance retires local patch 049 entirely.

## Bug status on main (verified 2026-07-16, `27ad9d1`)
Live: `model_executor/model_runner_components/load_model_utils.py:40`
```python
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing
```
used as a hard `datetime.timedelta` on the TP loading barrier (L266).

## PR title
```
[Fix] Make the unbalanced model-loading timeout overridable via env var
```

## PR body
```
### Motivation

The TP model-loading barrier hardcodes a 480 s timeout:

    UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480

Cold-cache loads of large quantized checkpoints on spinning-disk or
network-storage hosts routinely exceed it when ranks load unevenly (rank 0
reads 30+ GB while rank 1 waits at the barrier), killing the server mid-boot
with a distributed timeout even though the load would complete. Any fixed
constant is wrong for someone: raising the default punishes fast hosts with
slow failure detection; 480 s kills slow hosts that would succeed.

### Modification

    UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(
        os.environ.get("SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S", 480)
    )

Default behavior unchanged; affected deployments opt in with one env var.
(If maintainers prefer, happy to wire it through the envs registry instead —
say the word and I'll move it to sglang.srt.environ.)

### Validation

In production on our 2x RTX 3090 stack (TP=2, 24-30 GB AWQ checkpoints on
HDD-backed cold cache): first-boot-after-drop-caches exceeded 480 s and died
at the barrier; with the override (1800) every preset boots. Warm-cache boots
are unaffected.
```

## Checklist before opening
- [ ] Fork + branch `feat/load-timeout-env`
- [ ] Check whether main's envs registry (`sglang.srt.environ` /
      `envs.py`) is the preferred pattern now — if so submit in that idiom
- [ ] Run their pre-commit
