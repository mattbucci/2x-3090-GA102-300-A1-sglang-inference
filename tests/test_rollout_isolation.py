"""Invariants of the rollout network isolation (2026-09-19) and its auditor.

Run: python -m pytest -q tests/test_rollout_isolation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evals" / "swebench"))

import audit_leakage as al  # noqa: E402
import audit_predictions as ap  # noqa: E402
import docker_rollout as dr  # noqa: E402

SCAFFOLDS = ["opencode", "opencode-dcp", "little-coder", "little-coder-rtk", "prime", "dcode"]


def test_default_network_mode_is_none():
    import argparse
    p = dr.parse_args.__globals__["argparse"]  # sanity: module uses argparse
    assert p is argparse
    sys.argv = ["x", "--model", "sglang/q", "--out", "/tmp/x"]
    assert dr.parse_args().network_mode == "none"


def test_network_args_none_has_no_host_network(tmp_path):
    sock = tmp_path / "b.sock"
    args = dr.network_docker_args("none", sock, tmp_path)
    assert "--network=host" not in args
    assert args[:2] == ["--network", "none"]
    joined = " ".join(args)
    assert f"dst={dr.BRIDGE_SCRIPT_IN_CONTAINER},readonly" in joined
    assert f"src={sock},dst={dr.BRIDGE_SOCK_IN_CONTAINER}" in joined
    assert f"src={tmp_path},dst=/sessions" in joined


def test_network_args_host_is_explicit(tmp_path):
    args = dr.network_docker_args("host", None, tmp_path)
    assert args[0] == "--network=host"
    assert "/sessions" in " ".join(args)


def test_prelude_gates_on_bridge_and_strips_refs():
    pre = dr.isolation_prelude("none")
    assert f"container {dr.CONTAINER_SERVER_PORT} {dr.BRIDGE_SOCK_IN_CONTAINER}" in pre
    assert "BRIDGE CHECK FAILED" in pre and "exit 97" in pre
    assert "isolation: network=none bridge=127.0.0.1:23334" in pre
    assert "git -C /testbed tag -d" in pre and "update-ref -d" in pre and "remote remove" in pre
    assert 'isolation: refs=' in pre
    # host mode still strips refs (belt-and-braces) but never starts a bridge
    host = dr.isolation_prelude("host")
    assert "net_bridge" not in host and "update-ref -d" in host


def test_every_scaffold_has_one_diff_marker_for_the_snapshot():
    for sc in SCAFFOLDS:
        _envs, inner = dr.build_scaffold_invocation(sc, "sglang/q", "q", timeout=60, context_window=1000)
        assert inner.count("echo === DIFF ===") == 1, sc
        i = inner.index("echo === DIFF ===")
        assert i == 0 or inner[i - 1] == "\n", sc
        wired = inner.replace("echo === DIFF ===", dr.SESSION_SNAPSHOT + "echo === DIFF ===", 1)
        assert wired.index("snap .pi/agent/sessions") < wired.index("echo === DIFF ===")


def test_snapshot_skips_opencode_worktree_snapshots():
    s = dr.SESSION_SNAPSHOT
    assert "opencode.db" in s and "opencode/storage" in s
    assert "opencode/snapshot" not in s
    assert "/sessions/" in s


def test_bridge_failure_is_infra_not_model():
    log = "# stdout\n\n# stderr\nBRIDGE CHECK FAILED: server not reachable through the loopback bridge\n"
    cat, _ = ap.classify_log(log, 97, "", 3.0)
    assert cat == "infra_bridge"


def test_leak_classifier():
    inst = "django__django-11133"
    ev = al.classify_call(inst, "webfetch", {"url": "https://github.com/django/django/pull/11133.diff"}, "diff --git a/x", False)
    assert [(e["chan"], e["kind"], e["ok"]) for e in ev] == [("web", "UPSTREAM", True)]
    ev = al.classify_call(inst, "bash", {"command": "curl -sS https://github.com/django/django/pull/11133.diff"},
                          "curl: (6) Could not resolve host: github.com", None)
    assert ev[0]["kind"] == "UPSTREAM" and ev[0]["ok"] is False
    ev = al.classify_call(inst, "webfetch", {"url": "https://docs.python.org/3/library/re.html"}, "text", False)
    assert ev[0]["kind"] == "OTHER"
    ev = al.classify_call(inst, "websearch", {"query": "django 11133 HttpResponse memoryview"}, None, None)
    assert ev[0]["kind"] == "SEARCH" and ev[0]["ok"] is None
    ev = al.classify_call(inst, "bash", {"command": "git log --all -p -S memoryview -- django/http/response.py"}, "", None)
    assert [(e["chan"], e["kind"]) for e in ev] == [("git", "READ")]
    ev = al.classify_call(inst, "bash", {"command": "git log --oneline -5 && grep -rn foo django/"}, "abc", None)
    assert ev == []
    ev = al.classify_call(inst, "bash", {"command": "gh pr view 11133 --json body"}, "", None)
    assert ev[0]["kind"] == "UPSTREAM"


def test_isolation_proof_reads_log(tmp_path):
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "a__b-1.log").write_text(
        "# stdout\n=== DIFF ===\n\n# stderr\nisolation: network=none bridge=127.0.0.1:23334\nisolation: refs=1 tags=0\n")
    (run / "logs" / "a__b-2.log").write_text("# stdout\n\n# stderr\nisolation: refs=41 tags=120\n")
    assert al.isolation_proof(run, "a__b-1") == {"network_none": True, "refs_stripped": True}
    assert al.isolation_proof(run, "a__b-2") == {"network_none": False, "refs_stripped": False}
    rep = al.audit_run(run, require_isolation=True, gold={})
    assert rep["ok"] is False and rep["counts"]["iso_network_none"] == 1
