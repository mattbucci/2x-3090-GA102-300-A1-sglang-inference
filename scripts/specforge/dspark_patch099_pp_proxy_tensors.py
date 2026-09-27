#!/usr/bin/env python3
"""Patch 099 (test form): add the pp_proxy_tensors= kwarg the scheduler passes
to DSparkWorkerV2.forward_batch_generation and thread it to the target prefill
forward — mirrors dflash_worker_v2.py. v0.5.20 dies on the first DSPARK forward
without it (TypeError). Reversible: --revert restores the .bak.

R9700 6b59279 names this fix but did not push the patch file; verified against
/data/sglang-rebase-v0520 (worker sig had no pp_proxy_tensors, DFlash sibling did).
"""
import sys, shutil, pathlib
F = pathlib.Path("/data/sglang-rebase-v0520/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py")
bak = F.with_suffix(".py.pre099")

if "--revert" in sys.argv:
    if bak.exists():
        shutil.copy(bak, F); print("reverted from", bak)
    else:
        print("no backup to revert"); sys.exit(1)
    sys.exit(0)

src = F.read_text()
if "pp_proxy_tensors" in src:
    print("already patched (pp_proxy_tensors present) — no-op"); sys.exit(0)
shutil.copy(F, bak)

# 1) signature
old_sig = ("    def forward_batch_generation(\n"
           "        self,\n"
           "        batch: ScheduleBatch,\n"
           "        on_publish=None,\n"
           "        grammar_barrier=None,\n"
           "    ) -> GenerationBatchResult:\n")
new_sig = ("    def forward_batch_generation(\n"
           "        self,\n"
           "        batch: ScheduleBatch,\n"
           "        on_publish=None,\n"
           "        grammar_barrier=None,\n"
           "        pp_proxy_tensors=None,\n"
           "    ) -> GenerationBatchResult:\n")
assert src.count(old_sig) == 1, f"sig match count={src.count(old_sig)}"
src = src.replace(old_sig, new_sig)

# 2) forward to _forward_prefill with the kwarg
old_call = "            return self._forward_prefill(batch, on_publish)\n"
new_call = "            return self._forward_prefill(batch, on_publish, pp_proxy_tensors)\n"
assert src.count(old_call) == 1, f"prefill-call count={src.count(old_call)}"
src = src.replace(old_call, new_call)

# 3) _forward_prefill signature
old_pf = ("    def _forward_prefill(\n"
          "        self, batch: ScheduleBatch, on_publish\n"
          "    ) -> GenerationBatchResult:\n")
new_pf = ("    def _forward_prefill(\n"
          "        self, batch: ScheduleBatch, on_publish, pp_proxy_tensors=None\n"
          "    ) -> GenerationBatchResult:\n")
assert src.count(old_pf) == 1, f"prefill-sig count={src.count(old_pf)}"
src = src.replace(old_pf, new_pf)

# 4) thread into the two target_worker.forward_batch_generation calls in _forward_prefill
old_t1 = ("                self.target_worker.forward_batch_generation(\n"
          "                    batch, capture_hidden_mode=CaptureHiddenMode.FULL\n"
          "                )\n")
new_t1 = ("                self.target_worker.forward_batch_generation(\n"
          "                    batch,\n"
          "                    pp_proxy_tensors=pp_proxy_tensors,\n"
          "                    capture_hidden_mode=CaptureHiddenMode.FULL,\n"
          "                )\n")
assert src.count(old_t1) == 1, f"target-idle count={src.count(old_t1)}"
src = src.replace(old_t1, new_t1)

old_t2 = ("        batch_output = self.target_worker.forward_batch_generation(\n"
          "            batch, capture_hidden_mode=CaptureHiddenMode.FULL\n"
          "        )\n")
new_t2 = ("        batch_output = self.target_worker.forward_batch_generation(\n"
          "            batch,\n"
          "            pp_proxy_tensors=pp_proxy_tensors,\n"
          "            capture_hidden_mode=CaptureHiddenMode.FULL,\n"
          "        )\n")
assert src.count(old_t2) == 1, f"target-main count={src.count(old_t2)}"
src = src.replace(old_t2, new_t2)

F.write_text(src)
import py_compile; py_compile.compile(str(F), doraise=True)
print("patched + compiles OK; backup at", bak)
