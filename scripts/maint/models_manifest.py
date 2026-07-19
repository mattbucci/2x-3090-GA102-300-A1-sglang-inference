#!/usr/bin/env python3
"""Per-host manifest for the shared ~/AI/models directory (stdlib only, no GPU).

Walks every top-level entry, records what it *actually* is — resolved target,
real on-disk size, shard count, quantization, mtimes, and which repo checkouts
reference the name — then flags three failure classes:

  TRAP       a name that presents as a BF16 base (carries a `bf16` marker and
             *no* quant-format marker) but resolves to quantized weights.  Feeding
             such a name as a calibration base silently double-quantizes the model
             (the 16h-loss class).  Honest names that advertise the quant stage
             (e.g. `...-BF16-v1-AWQ-CT`) are NOT traps — the marker guard excludes
             them.
  COLLISION  two distinct entries whose names are equal under casefold() (cannot
             coexist on a case-insensitive fs, e.g. APFS if ever synced to a Mac).
  DANGLING   a symlink whose resolved target does not exist.

Fleet-audit lane 3090-H.  See experiments/04-models-manifest-bf16-symlink-defuse.md
and rules-for-agents.md "Host filesystem layout".

Usage:
    python scripts/maint/models_manifest.py \
        --models-dir ~/AI/models \
        --out ~/AI/models/MANIFEST.json \
        --md  ~/AI/models/MANIFEST.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

# A name is "claiming BF16 semantics" only if it carries a bf16 marker AND no
# quant-format marker.  If it also advertises a quant format, the name is honest
# about being quantized and must not be flagged TRAP.
_BF16_MARK = re.compile(r"(?i)bf16")
_QUANT_MARK = re.compile(
    r"(?i)(awq|gptq|int4|fp8|w4a16|rtn|marlin|autoround|gguf|nf4|(^|[-_])ct([-_]|$))"
)


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _du_bytes(path):
    """Real on-disk size of the resolved target (follows symlinks: du -sbL)."""
    try:
        out = subprocess.run(
            ["du", "-sbL", path], capture_output=True, text=True, timeout=900
        )
        if out.returncode == 0 and out.stdout.split():
            return int(out.stdout.split()[0])
    except Exception:
        pass
    return None


def _load_config(realpath):
    cfg = os.path.join(realpath, "config.json")
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg) as f:
            return json.load(f)
    except Exception:
        return {}  # present but unparseable — treated as "has a config, quant unknown"


def _quant_of(config):
    """Returns (quant_label, quant_config_or_None, torch_dtype)."""
    if config is None:
        return "none/BF16", None, None
    qc = config.get("quantization_config")
    dtype = config.get("torch_dtype")
    if not qc:
        return "none/BF16", None, dtype
    method = qc.get("quant_method")
    bits = qc.get("bits")
    return (f"{method}-{bits}bit" if bits else str(method)), qc, dtype


def _shard_count(realpath):
    idx = os.path.join(realpath, "model.safetensors.index.json")
    if os.path.isfile(idx):
        try:
            with open(idx) as f:
                return len(set(json.load(f).get("weight_map", {}).values()))
        except Exception:
            return None
    for single in ("model.safetensors", "pytorch_model.bin", "consolidated.safetensors"):
        if os.path.isfile(os.path.join(realpath, single)):
            return 1
    return 0  # no weights found (intermediate / non-model dir)


def _find_checkouts(ai_root):
    """Best-effort: sibling repo checkouts under ~/AI that contain a scripts/ dir."""
    out = []
    try:
        for name in sorted(os.listdir(ai_root)):
            d = os.path.join(ai_root, name)
            if os.path.isdir(d) and os.path.isdir(os.path.join(d, "scripts")):
                out.append(d)
    except Exception:
        pass
    return out


def _consumers(name, checkouts):
    """grep -rlF the entry name across checkouts (best-effort, text files only)."""
    hits = []
    for repo in checkouts:
        try:
            out = subprocess.run(
                ["grep", "-rlF", "--exclude-dir=.git",
                 "--include=*.py", "--include=*.sh", "--include=*.md",
                 "--include=*.json", "--include=*.jinja", "--include=*.txt",
                 "--", name, repo],
                capture_output=True, text=True, timeout=120,
            )
            for line in out.stdout.splitlines():
                hits.append(os.path.relpath(line, os.path.dirname(repo)))
        except Exception:
            pass
    return sorted(set(hits))


def build(models_dir, ai_root=None):
    models_dir = os.path.abspath(os.path.expanduser(models_dir))
    if ai_root is None:
        ai_root = os.path.dirname(models_dir)  # ~/AI for ~/AI/models
    checkouts = _find_checkouts(ai_root)

    entries = []
    for name in sorted(os.listdir(models_dir)):
        full = os.path.join(models_dir, name)
        is_link = os.path.islink(full)
        realpath = os.path.realpath(full)
        target_exists = os.path.exists(realpath)
        config = _load_config(realpath) if target_exists else None
        quant, qc, dtype = _quant_of(config)

        flags = []
        if _BF16_MARK.search(name) and not _QUANT_MARK.search(name) and qc is not None:
            flags.append("TRAP")
        if is_link and not target_exists:
            flags.append("DANGLING")

        try:
            lmtime = _iso(os.lstat(full).st_mtime)
        except Exception:
            lmtime = None
        tmtime = _iso(os.stat(realpath).st_mtime) if target_exists else None

        entries.append({
            "name": name,
            "entry_type": "symlink" if is_link else "dir",
            "target": realpath,
            "target_exists": target_exists,
            "bytes": _du_bytes(realpath) if target_exists else 0,
            "mtime": lmtime,            # the entry's own mtime (link mtime for symlinks)
            "target_mtime": tmtime,     # resolved target mtime (differs → link predates base)
            "shard_count": _shard_count(realpath) if target_exists else None,
            "quant": quant,
            "torch_dtype": dtype,
            "consumers": _consumers(name, checkouts),
            "flags": flags,
        })

    # COLLISION pass across all names (distinct names sharing a casefold key)
    by_fold = defaultdict(set)
    for e in entries:
        by_fold[e["name"].casefold()].add(e["name"])
    for e in entries:
        if len(by_fold[e["name"].casefold()]) > 1:
            e["flags"].append("COLLISION")

    return entries, checkouts, models_dir


def df_of(path):
    try:
        out = subprocess.run(
            ["df", "-B1", "--output=source,fstype,size,used,avail,pcent,target", path],
            capture_output=True, text=True, timeout=30,
        )
        rows = out.stdout.strip().splitlines()
        if len(rows) >= 2:
            k = rows[0].split()
            v = rows[1].split()
            return dict(zip(k, v))
    except Exception:
        pass
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="~/AI/models")
    ap.add_argument("--out", default="~/AI/models/MANIFEST.json")
    ap.add_argument("--md", default="~/AI/models/MANIFEST.md")
    ap.add_argument("--ai-root", default=None, help="override checkout scan root")
    args = ap.parse_args()

    entries, checkouts, models_dir = build(args.models_dir, args.ai_root)
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    host = socket.gethostname()
    real_root = os.path.realpath(models_dir)
    disk = df_of(models_dir)

    flag_counts = defaultdict(int)
    for e in entries:
        for fl in e["flags"]:
            flag_counts[fl] += 1
    total_bytes = sum(e["bytes"] or 0 for e in entries)

    header = {
        "host": host,
        "generated": now,
        "models_dir": models_dir,
        "resolved_root": real_root,
        "models_dir_is_symlink": os.path.islink(models_dir),
        "df": disk,
        "checkouts_scanned": checkouts,
        "entry_count": len(entries),
        "total_bytes": total_bytes,
        "flag_counts": dict(flag_counts),
    }
    doc = {"header": header, "entries": entries}

    out = os.path.expanduser(args.out)
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)

    def gib(b):
        return f"{(b or 0) / 1024**3:.1f}G"

    lines = [
        f"# ~/AI/models manifest — {host}",
        "",
        f"- Generated: {now}",
        f"- models_dir: `{models_dir}`"
        + (f" → `{real_root}`" if header["models_dir_is_symlink"] else " (real dir)"),
        f"- Filesystem: {disk.get('Source','?')} {disk.get('Fstype','')} "
        f"{gib(int(disk.get('1B-blocks',0) or 0))} total, "
        f"{gib(int(disk.get('Avail',0) or 0))} free ({disk.get('Use%','?')} used) "
        f"on {disk.get('Mounted','?')}",
        f"- Entries: {len(entries)} · {gib(total_bytes)} resolved · "
        f"flags: {dict(flag_counts) or 'none'}",
        f"- Checkouts scanned for consumers: "
        + (", ".join(f"`{os.path.basename(c)}`" for c in checkouts) or "none"),
        "",
        "| Entry | Type | Size | Quant | Shards | Flags | Target |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in sorted(entries, key=lambda x: (not x["flags"], -(x["bytes"] or 0))):
        tgt = e["target"] if e["entry_type"] == "symlink" else ""
        lines.append(
            f"| `{e['name']}` | {e['entry_type']} | {gib(e['bytes'])} | "
            f"{e['quant']} | {e['shard_count']} | "
            f"{' '.join(e['flags']) or '—'} | "
            f"{('`'+tgt+'`') if tgt else ''} |"
        )
    lines.append("")
    md = os.path.expanduser(args.md)
    with open(md, "w") as f:
        f.write("\n".join(lines))

    # stdout summary (self-test surface)
    print(f"host={host}  entries={len(entries)}  total={gib(total_bytes)}")
    print(f"flags: {dict(flag_counts) or 'none'}")
    for e in entries:
        if e["flags"]:
            print(f"  [{' '.join(e['flags'])}] {e['name']} -> {e['target']} ({e['quant']})")
    print(f"wrote {out}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
