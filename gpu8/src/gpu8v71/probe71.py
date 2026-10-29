"""V7.1 section 15 — local admission probe on the real feature pack.

Full gene background x 16 real diseases, image branch on, three forward +
backward updates on CUDA; records the wall time of each update, the peak
process RSS (10 ms sampling) and the GPU allocated / reserved peaks.  If CUDA
is requested but unavailable the probe stops (it never silently falls back to
the CPU and still calls itself a GPU test).
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import psutil
import torch

from . import paths71
from .common71 import dump_json, now_utc
from .core71 import CoreConfig, TypedCore
from .featurepack71 import FeaturePack
from .train71 import D_KEYS, take


def run(device: str = "cuda", n_dis: int = 16) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    proc = psutil.Process()
    peak = {"rss": proc.memory_info().rss}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            time.sleep(0.01)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    pack = FeaturePack()
    T = pack.tensors(device, image=True)
    torch.cuda.reset_peak_memory_stats()
    m = TypedCore(CoreConfig(image=True)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    rows = torch.arange(n_dis, device=device) * 97 % pack.n_d
    g = {k: T[k] for k in ("xS", "mS", "xF", "mF", "e", "o", "xI", "mI")}
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        out = m(take(T, D_KEYS, rows), g)
        loss = -(out["r"] * out["valid"]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    stop.set()
    th.join()
    grads = {n: float(p.grad.abs().sum()) if p.grad is not None else None
             for n, p in m.named_parameters()}
    rep = {"generated_utc": now_utc(), "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "genes": pack.n_g, "diseases": n_dis, "image": True,
           "trainable_parameters": m.n_trainable(), "update_seconds": times,
           "peak_rss_mib": peak["rss"] / 2 ** 20,
           "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
           "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
           "gradient_abs_sum": grads,
           "note": "real feature pack; a connection/resource probe, not a performance result"}
    dump_json(paths71.AUDIT / "cuda_probe_v71.json", rep)
    return rep


if __name__ == "__main__":
    paths71.ensure()
    print(json.dumps(run(), indent=2))
