#!/usr/bin/env python3
"""Plain-PyTorch reference forward for Flint (RecursiveSI/flint), layer-streamed from safetensors.

Runs the BF16 base OR a native-AWQ INT4 checkpoint (dequantized exactly, via the quantizer's
`unpack_awq`) in fp32 on CPU or one GPU, over one fixed token-id prompt, and writes per-position
top-20 logprobs + the logprob of the actual next token — the same shape `served_prompt_logprobs.py`
collects from a running server. Purpose: a serving-stack-independent reference, so that

  BF16 ref  vs  INT4-dequant ref   = the quantization error alone
  INT4-dequant ref  vs  served INT4 = the serving path alone (triton attention + sinks + SWA, AWQ-Marlin MoE)

Model semantics follow flint_sglang/model.py: per-head QK RMSNorm, neox RoPE (theta 10000) on the
sliding layers only, no positional encoding on `no_rope_layer_indices`, sliding window = the current
token + 512 preceding, learned per-head sinks = one extra softmax logit with no value, sigmoid top-6
routing renormalized to sum 1 (expert bias 0), an ungated shared expert, dense SwiGLU on layers 0-1.

Usage:
  # 1. token ids: render a committed replay prefix through the checkpoint's chat template
  flint_cpu_reference.py --model <bf16-dir> --replay benchmarks/quality/flint/replay/replay-12907-k4.json.gz \
      --max-tokens 2048 --save-ids ids.json --ids-only
  # 2. reference logprobs (GPU: ~2 min for 2048 tokens incl. the 80 GB weight stream; CPU: hours)
  flint_cpu_reference.py --model <bf16-dir> --ids ids.json --device cuda:0 --out ref-bf16.json
  flint_cpu_reference.py --model <awq-dir>  --ids ids.json --device cuda:0 --out ref-int4.json
  # 3. served side + comparison
  served_prompt_logprobs.py ids.json served.json ; compare_prompt_logprobs.py ref-bf16.json ref-int4.json served.json
"""
import argparse, gzip, json, math, os, re, sys, time
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantize"))
from quantize_flint_rtn_awq import unpack_awq  # noqa: E402


class Weights:
    """name -> fp32 tensor, from a BF16 or native-AWQ checkpoint (dequantized on read)."""

    def __init__(self, d: str, device: str, overlay: "Weights | None" = None, overlay_re: str | None = None):
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.d, self.device, self.map, self.handles = d, device, idx, {}
        self.names = set(idx)
        # hybrid reference: names matching overlay_re are read from the overlay checkpoint instead
        # (e.g. BF16 base + INT4-dequantized experts only) -> attributes quantization error per module
        self.overlay, self.overlay_re = overlay, (re.compile(overlay_re) if overlay_re else None)
        self.overlay_hits = 0

    def _raw(self, name):
        fn = self.map[name]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.d, fn), "pt", device="cpu")
        return self.handles[fn].get_tensor(name)

    def get(self, name):
        if self.overlay_re is not None and self.overlay_re.search(name):
            self.overlay_hits += 1
            return self.overlay.get(name)
        if name in self.names:
            return self._raw(name).to(self.device, torch.float32)
        base = name[: -len(".weight")]
        if base + ".qweight" in self.names:
            w = unpack_awq(self._raw(base + ".qweight"), self._raw(base + ".qzeros"), self._raw(base + ".scales"))
            return w.to(self.device, torch.float32)
        raise KeyError(name)


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def render_ids(model_dir, replay, max_tokens):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    r = json.load(gzip.open(replay) if replay.endswith(".gz") else open(replay))
    msgs = json.loads(json.dumps(r["messages"]))
    for m in msgs:  # the server parses OpenAI argument strings into dicts before rendering
        for tc in m.get("tool_calls") or []:
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                try:
                    tc["function"]["arguments"] = json.loads(a)
                except json.JSONDecodeError:
                    tc["function"]["arguments"] = {"raw": a}
    text = tok.apply_chat_template(msgs, tools=r.get("tools"), add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text)
    print(f"rendered {replay}: {len(ids)} tokens (session recorded {r.get('input_tokens')}); using first {max_tokens}")
    return ids[:max_tokens]


@torch.no_grad()
def forward(W: Weights, cfg: dict, ids, device, topk=20, log_every=5, n_layers=None):
    """Prompt logprobs over `ids`; n_layers (debug) stops early and returns {"hidden": x} instead."""
    T = len(ids)
    H, nh, nkv, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps, L = cfg["rms_norm_eps"], cfg["num_hidden_layers"]
    nope, dense = set(cfg["no_rope_layer_indices"]), set(cfg["mlp_only_layers"])
    win = cfg["sliding_window"] - 1  # preceding tokens visible besides the current one
    scale = hd ** -0.5
    t_ids = torch.tensor(ids, device=device)
    x = W.get("model.embed_tokens.weight")[t_ids]
    # neox RoPE tables
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    fr = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv[None]
    emb = torch.cat([fr, fr], -1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]

    def rope(v):  # [T, heads, hd]
        v1, v2 = v[..., : hd // 2], v[..., hd // 2 :]
        return v * cos + torch.cat([-v2, v1], -1) * sin

    pos = torch.arange(T, device=device)
    causal = pos[None, :] <= pos[:, None]
    swa = causal & ((pos[:, None] - pos[None, :]) <= win)
    t0 = time.time()
    for l in range(L):
        p = f"model.layers.{l}."
        sliding = cfg["layer_types"][l] == "sliding_attention"
        h = rmsnorm(x, W.get(p + "input_layernorm.weight"), eps)
        q = (h @ W.get(p + "self_attn.q_proj.weight").T).view(T, nh, hd)
        k = (h @ W.get(p + "self_attn.k_proj.weight").T).view(T, nkv, hd)
        v = (h @ W.get(p + "self_attn.v_proj.weight").T).view(T, nkv, hd)
        q = rmsnorm(q, W.get(p + "self_attn.q_norm.weight"), eps)
        k = rmsnorm(k, W.get(p + "self_attn.k_norm.weight"), eps)
        if l not in nope:
            q, k = rope(q), rope(k)
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)  # q head i uses kv head i // rep
        v = v.repeat_interleave(rep, dim=1)
        s = torch.einsum("thd,shd->hts", q, k) * scale
        s = s.masked_fill(~(swa if sliding else causal)[None], float("-inf"))
        sink = W.get(p + "self_attn.softmax_offset")  # [nh], logit units
        s = torch.cat([s, sink[:, None, None].expand(nh, T, 1)], -1)
        pr = torch.softmax(s, -1)[..., :T]
        o = torch.einsum("hts,shd->thd", pr, v).reshape(T, nh * hd)
        x = x + o @ W.get(p + "self_attn.o_proj.weight").T
        h = rmsnorm(x, W.get(p + "post_attention_layernorm.weight"), eps)
        if l in dense:
            g = h @ W.get(p + "mlp.gate_proj.weight").T
            u = h @ W.get(p + "mlp.up_proj.weight").T
            m = (torch.nn.functional.silu(g) * u) @ W.get(p + "mlp.down_proj.weight").T
        else:
            logits = h @ W.get(p + "mlp.gate.weight").T
            sc = torch.sigmoid(logits)
            top_i = torch.topk(sc, cfg["num_experts_per_tok"], dim=-1, sorted=False).indices
            top_w = sc.gather(1, top_i)
            top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20)
            sg = h @ W.get(p + "mlp.shared_expert.gate_proj.weight").T
            su = h @ W.get(p + "mlp.shared_expert.up_proj.weight").T
            m = (torch.nn.functional.silu(sg) * su) @ W.get(p + "mlp.shared_expert.down_proj.weight").T
            for e in torch.unique(top_i).tolist():
                rows, slot = (top_i == e).nonzero(as_tuple=True)
                he = h[rows]
                pe = p + f"mlp.experts.{e}."
                ge = he @ W.get(pe + "gate_proj.weight").T
                ue = he @ W.get(pe + "up_proj.weight").T
                ye = (torch.nn.functional.silu(ge) * ue) @ W.get(pe + "down_proj.weight").T
                m.index_add_(0, rows, ye * top_w[rows, slot][:, None])
        x = x + m
        if (l + 1) % log_every == 0 or l == L - 1:
            print(f"  layer {l + 1}/{L} {time.time() - t0:.0f}s", flush=True)
        if n_layers is not None and l + 1 >= n_layers:
            return {"hidden": x}
    x = rmsnorm(x, W.get("model.norm.weight"), eps)
    lp = torch.log_softmax(x @ W.get("lm_head.weight").T, -1)  # [T, V]
    tv, ti = lp.topk(topk, dim=-1)
    nxt = lp[torch.arange(T - 1, device=device), t_ids[1:]]
    ent = -(lp.exp() * lp).sum(-1)
    return {
        "next_lp": nxt.tolist() + [None],
        "top": [[(int(i), float(v)) for i, v in zip(ti[t].tolist(), tv[t].tolist())] for t in range(T)],
        "entropy": ent.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="BF16 base dir or native-AWQ checkpoint dir")
    ap.add_argument("--ids", help="ids.json {input_ids:[...]}")
    ap.add_argument("--replay", help="replay prefix (.json/.json.gz) to render through the chat template")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--save-ids", help="write the rendered ids here")
    ap.add_argument("--ids-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", help="output json")
    ap.add_argument("--overlay", help="second checkpoint dir (e.g. native-AWQ) whose tensors replace --model's for names matching --overlay-re")
    ap.add_argument("--overlay-re", help=r"regex on tensor names, e.g. 'mlp\.experts\.' or 'self_attn\.'")
    args = ap.parse_args()
    if args.ids:
        ids = json.load(open(args.ids))["input_ids"]
    elif args.replay:
        ids = render_ids(args.model, args.replay, args.max_tokens)
    else:
        ap.error("--ids or --replay")
    if args.save_ids:
        json.dump({"input_ids": ids, "source": args.replay or args.ids}, open(args.save_ids, "w"))
    if args.ids_only:
        return
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    ov = Weights(args.overlay, args.device) if args.overlay else None
    W = Weights(args.model, args.device, ov, args.overlay_re)
    t0 = time.time()
    res = forward(W, cfg, ids, args.device)
    res.update({"input_ids": ids, "model": args.model, "device": args.device, "secs": round(time.time() - t0, 1),
                "quantized": any(n.endswith(".qweight") for n in W.names),
                "overlay": args.overlay, "overlay_re": args.overlay_re, "overlay_hits": W.overlay_hits})
    json.dump(res, open(args.out, "w"))
    print(f"FLINT_REF_DONE {args.model} T={len(ids)} {res['secs']}s -> {args.out}")


if __name__ == "__main__":
    main()
