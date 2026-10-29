"""Hardware probe: verify the five preconditions before any research run.

Plan section 4.3.  Records the device, runs a real forward and backward pass of
the actual model, times a chunked scoring pass and a ridge solve, and reports
peak memory.  Asking for CUDA when CUDA is unavailable is an error, never a
silent fall back to CPU pretending to be a GPU measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from gpu8 import adapt as A
from gpu8 import paths
from gpu8.config import load_config
from gpu8.model import build_model
from gpu8.resources import ResourceMonitor, host_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GPU8 hardware and model probe.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--score-batch", type=int, default=8192)
    ap.add_argument("--n-genes", type=int, default=20000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--run-context", default="user_local")
    ap.add_argument("--output", default="")
    args = ap.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but torch.cuda.is_available() is False", file=sys.stderr)
        return 2
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device

    cfg = load_config()
    model = build_model(cfg.model, seed=42).to(device)
    n_param = model.n_trainable()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    out: dict = {"run_context": args.run_context, "device": device,
                 "requested_device": args.device, "amp": bool(args.amp),
                 "trainable_parameters": n_param,
                 "parameter_breakdown": model.module_totals(),
                 "config_hash": cfg.config_hash, "host": host_report()}

    g = torch.Generator(device="cpu").manual_seed(7)

    def batch(n):
        return (torch.randn(n, cfg.model.node_in_dim, generator=g).to(device),
                torch.randn(n, cfg.model.node_in_dim, generator=g).to(device),
                torch.randn(n, 4, generator=g).to(device),
                (torch.rand(n, 4, generator=g) > 0.15).float().to(device),
                torch.randn(n, 12, generator=g).to(device))

    # ---- synthetic training steps ---------------------------------------
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device == "cuda") else None
    n_per_task = 5 + cfg.candidates.n_anchors + cfg.train.max_query_pos_per_task \
        + cfg.train.max_query_anchor_per_task
    with ResourceMonitor(interval=0.02, sample_nvidia=(device == "cuda")) as mon:
        t0 = time.perf_counter()
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(args.episodes):
                args_b = batch(n_per_task)
                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        enc = model.encode(*args_b)
                    base = enc.base_score.float()
                    phi = enc.phi.float()
                else:
                    enc = model.encode(*args_b)
                    base, phi = enc.base_score, enc.phi
                k, u = 5, cfg.candidates.n_anchors
                w = torch.softmax(torch.randn(k, device=device), 0)
                summary = torch.rand(4, device=device)
                lam, rho = model.controls(summary)
                res = A.adapt(phi[:k], phi[k:k + u], base[:k], base[k:k + u], w,
                              lam, rho, b_clip=cfg.model.ridge_b_clip)
                s = A.apply_adaptation(base, phi, res.a)
                qp = s[k + u:k + u + cfg.train.max_query_pos_per_task]
                qa = s[k + u + cfg.train.max_query_pos_per_task:]
                loss = torch.nn.functional.softplus(-(qp.unsqueeze(1) - qa.unsqueeze(0))).mean()
                (loss / args.episodes).backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
    out["train"] = {"steps": args.steps, "episodes_per_step": args.episodes,
                    "pairs_per_episode": n_per_task, "seconds": round(train_s, 4),
                    "seconds_per_step": round(train_s / max(1, args.steps), 5)}

    # ---- chunked scoring -------------------------------------------------
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        done = 0
        while done < args.n_genes:
            n = min(args.score_batch, args.n_genes - done)
            enc = model.encode(*batch(n))
            _ = A.apply_adaptation(enc.base_score, enc.phi,
                                   torch.zeros(cfg.model.adapt_dim, device=device))
            done += n
        if device == "cuda":
            torch.cuda.synchronize()
        score_s = time.perf_counter() - t0
    out["score"] = {"n_pairs": args.n_genes, "batch": args.score_batch,
                    "seconds": round(score_s, 4)}

    # ---- ridge solve timing ---------------------------------------------
    phi_s = torch.randn(5, cfg.model.adapt_dim, device=device)
    phi_a = torch.randn(cfg.candidates.n_anchors, cfg.model.adapt_dim, device=device)
    phi_s = phi_s / torch.clamp(phi_s.norm(dim=-1, keepdim=True), min=1.0)
    phi_a = phi_a / torch.clamp(phi_a.norm(dim=-1, keepdim=True), min=1.0)
    bs, ba = torch.randn(5, device=device), torch.randn(cfg.candidates.n_anchors, device=device)
    w = torch.softmax(torch.randn(5, device=device), 0)
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(200):
            A.adapt(phi_s, phi_a, bs, ba, w, torch.tensor(0.7, device=device),
                    torch.tensor(1.0, device=device))
        if device == "cuda":
            torch.cuda.synchronize()
        solve_s = (time.perf_counter() - t0) / 200
    out["ridge_solve"] = {"design_shape": [5 * cfg.candidates.n_anchors, cfg.model.adapt_dim],
                          "system_shape": [cfg.model.adapt_dim, cfg.model.adapt_dim],
                          "seconds_per_solve": round(solve_s, 8)}

    r = mon.result
    out["resources"] = {"peak_rss_mib": round(r.peak_rss_mib, 1),
                        "peak_gpu_allocated_mib": round(r.peak_gpu_allocated_mib, 1),
                        "peak_gpu_reserved_mib": round(r.peak_gpu_reserved_mib, 1),
                        "nvidia_peak_used_mib": round(r.nvidia_peak_used_mib, 1),
                        "samples": r.samples}
    out["caveat"] = ("synthetic tensors: this measures implementation feasibility and "
                     "kernel cost, not paper accuracy and not the cost of real feature "
                     "construction, candidate retrieval or baseline training")

    dest = Path(args.output) if args.output else \
        paths.RESOURCE_LOG_DIR / f"probe_{device}{'_amp' if args.amp else '_fp32'}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("device", "trainable_parameters", "train", "score",
                       "ridge_solve", "resources")}, indent=2, default=str))
    print(f"written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
