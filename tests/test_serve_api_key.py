#!/usr/bin/env python3
"""Offline checks that the bake-off's server credential reaches every scaffold.

SERVE_MODE=docker serves from the OCI image behind secure-launch, which requires
an API key; serve_backend.sh mints one per cycle and points
SWEBENCH_API_KEY_FILE at it. Every scaffold's provider config (and our own
preflight) must carry that exact value, and bare metal (no file) must keep the
historical `noop` placeholder. Runs without docker or a GPU:
    python tests/test_serve_api_key.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evals" / "swebench"))
import docker_rollout as dr  # noqa: E402

SCAFFOLDS = ("opencode", "opencode-dcp", "little-coder", "little-coder-rtk", "prime", "dcode", "claw-code")


def _material(scaffold: str) -> str:
    envs, inner = dr.build_scaffold_invocation(scaffold, "sglang/qwen38", "qwen38",
                                               timeout=1800, context_window=262144)
    return " ".join(envs) + "\n" + inner


class ApiKeyPlumbing(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in ("SWEBENCH_API_KEY_FILE", "SWEBENCH_API_KEY")}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_bare_metal_keeps_placeholder(self):
        self.assertEqual(dr.api_key(), dr.API_KEY_PLACEHOLDER)
        self.assertFalse(dr.api_auth_enabled())
        self.assertNotIn("Authorization", dr._auth_headers())
        for sc in SCAFFOLDS:
            self.assertIn(dr.API_KEY_PLACEHOLDER, _material(sc), sc)

    def test_key_file_reaches_every_scaffold(self):
        key = "k" * 48
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "api-key"
            p.write_text(key + "\n")
            os.environ["SWEBENCH_API_KEY_FILE"] = str(p)
            self.assertEqual(dr.api_key(), key)
            self.assertTrue(dr.api_auth_enabled())
            self.assertEqual(dr._auth_headers()["Authorization"], f"Bearer {key}")
            for sc in SCAFFOLDS:
                m = _material(sc)
                self.assertIn(key, m, sc)
                self.assertNotIn(dr.API_KEY_PLACEHOLDER, m, sc)

    def test_empty_key_file_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "api-key"
            p.write_text("\n")
            os.environ["SWEBENCH_API_KEY_FILE"] = str(p)
            with self.assertRaises(ValueError):
                dr.api_key()


class ServeBackendScript(unittest.TestCase):
    """serve_backend.sh in bare mode needs no docker; docker mode must mint a
    key pair secure-launch accepts (32-512 printable ASCII bytes, one line,
    not group/other-writable) and export it for docker_rollout."""

    def _run(self, body: str, env: dict) -> subprocess.CompletedProcess:
        script = (f"source {REPO}/scripts/common.sh\n"
                  f"source {REPO}/evals/swebench/serve_backend.sh\n" + body)
        return subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                              env={**os.environ, **env})

    def test_bare_mode_exports_nothing(self):
        r = self._run('serve_backend_init && echo "mode=$SERVE_MODE key=${SWEBENCH_API_KEY_FILE:-unset}"',
                      {"SERVE_MODE": "bare"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("mode=bare key=unset", r.stdout)

    def test_bad_mode_rejected(self):
        r = self._run("serve_backend_init", {"SERVE_MODE": "podman"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("must be bare or docker", r.stderr)

    def test_tracked_default_is_a_valid_mode(self):
        conf = (REPO / "evals" / "swebench" / "serve_mode.conf").read_text().split()[0]
        self.assertIn(conf, ("bare", "docker"))

    def test_docker_mode_mints_acceptable_keys(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._run('_serve_mint_keys && echo "$SWEBENCH_API_KEY_FILE"',
                          {"SERVE_MODE": "docker", "LOG_DIR": d})
            self.assertEqual(r.returncode, 0, r.stderr)
            key_file = Path(r.stdout.strip())
            self.assertEqual(key_file, Path(d) / "secrets" / "api-key")
            self.assertEqual(oct(key_file.parent.stat().st_mode & 0o777), "0o700")
            for name in ("api-key", "admin-key"):
                f = key_file.parent / name
                raw = f.read_bytes()
                self.assertEqual(f.stat().st_mode & 0o022, 0, name)
                self.assertTrue(raw.endswith(b"\n") and raw.count(b"\n") == 1, name)
                v = raw[:-1]
                self.assertTrue(32 <= len(v) <= 512, name)
                self.assertTrue(all(33 <= b <= 126 for b in v), name)
            self.assertNotEqual((key_file.parent / "api-key").read_text(),
                                (key_file.parent / "admin-key").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
