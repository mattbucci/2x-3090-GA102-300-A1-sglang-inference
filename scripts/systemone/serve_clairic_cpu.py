#!/usr/bin/env python3
"""CPU System One decision server for Cloudflare/clairic-flash (default port 30001).

Clairic Flash is a Qwen3.5-9B backbone plus a structured-output head and its own Python
runtime (`clairic_model.systemone`); it is not an SGLang model, so it is served here with the
release's Transformers runtime on the CPU, leaving both GPUs to SGLang. The endpoints match the
System One decision-server contract the router expects (simple-llm-router ADR-0023):

  POST /v1/systemone   {"model": ..., "state": ..., "fields": {...}, "images": [...], "options": {...}}
  GET  /v1/models      {"object": "list", "data": [{"id": <model id>, ...}]}
  GET  /health         200 once the model is loaded

There is no chat route. Requests run one at a time (the release runtime is single-concurrency).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", default="/data/models/hf-3rdparty/clairic-flash")
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=30001)
ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
ap.add_argument("--threads", type=int, default=int(os.environ.get("CLAIRIC_THREADS", "12")))
ap.add_argument("--max-length", type=int, default=65273)
args = ap.parse_args()

torch.set_num_threads(args.threads)
sys.path.insert(0, args.model_path)
from clairic_model import load_release_model, systemone  # noqa: E402

t0 = time.time()
MODEL, PROCESSOR = load_release_model(
    args.model_path, device="cpu", dtype=getattr(torch, args.dtype), max_length=args.max_length
)
MODEL_ID = MODEL.runtime.settings.model_id
print(f"loaded {MODEL_ID} on cpu ({args.dtype}, {args.threads} threads) in {time.time() - t0:.0f} s", flush=True)
LOCK = threading.Lock()
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "cloudflare",
                                        "max_model_len": args.max_length}]}


@app.post("/v1/systemone")
def predict(body: dict):
    # A plain def runs in FastAPI's worker thread, outside the event loop, which the release
    # runtime's synchronous predict() requires.
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    name = body.pop("model", None)
    if name is not None and name != MODEL_ID:
        raise HTTPException(404, f"model {name!r} is not served here (serving {MODEL_ID!r})")
    body.pop("plugins", None)
    try:
        with LOCK:
            out = systemone(MODEL, PROCESSOR, body)
    except Exception as e:  # validation errors from the release runtime are the caller's fault
        name_ = type(e).__name__
        status = 400 if name_ in ("ValidationError", "ValueError", "PydanticCustomError") else 500
        return JSONResponse({"error": {"type": name_, "message": str(e)[:2000]}}, status_code=status)
    if isinstance(out, dict):
        out.setdefault("model", MODEL_ID)
    return out


uvicorn.run(app, host=args.host, port=args.port, log_level="info")
