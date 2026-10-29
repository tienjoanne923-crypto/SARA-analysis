"""Resource accounting: wall clock, peak RSS, GPU allocated/reserved.

The research plan requires every formal run to report real timing and memory,
and explicitly forbids inferring resource claims from parameter counts.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:  # psutil is optional at import time so audits can run without it
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _nvidia_smi_query() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return {}
        first = out.stdout.strip().splitlines()[0].split(",")
        return {
            "name": first[0].strip(),
            "memory_total_mib": float(first[1]),
            "memory_used_mib": float(first[2]),
            "driver_version": first[3].strip(),
        }
    except Exception:
        return {}


def host_report() -> dict:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }
    if psutil is not None:
        info["ram_total_gib"] = round(psutil.virtual_memory().total / (1 << 30), 2)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    info["nvidia_smi"] = _nvidia_smi_query()
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["cuda_total_memory_gib"] = round(props.total_memory / (1 << 30), 2)
            info["cuda_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:  # pragma: no cover
        info["torch_import_error"] = str(exc)
    return info


@dataclass
class ResourceSample:
    peak_rss_mib: float = 0.0
    peak_gpu_allocated_mib: float = 0.0
    peak_gpu_reserved_mib: float = 0.0
    seconds: float = 0.0
    samples: int = 0
    nvidia_peak_used_mib: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class ResourceMonitor:
    """Samples RSS / GPU memory every `interval` seconds inside a `with` block."""

    def __init__(self, interval: float = 0.25, sample_nvidia: bool = False):
        self.interval = interval
        self.sample_nvidia = sample_nvidia
        self.result = ResourceSample()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    # -- sampling ---------------------------------------------------------
    def _sample_once(self) -> None:
        if psutil is not None:
            try:
                proc = psutil.Process()
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except Exception:
                        pass
                self.result.peak_rss_mib = max(self.result.peak_rss_mib, rss / (1 << 20))
            except Exception:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                self.result.peak_gpu_allocated_mib = max(
                    self.result.peak_gpu_allocated_mib,
                    torch.cuda.max_memory_allocated() / (1 << 20))
                self.result.peak_gpu_reserved_mib = max(
                    self.result.peak_gpu_reserved_mib,
                    torch.cuda.max_memory_reserved() / (1 << 20))
        except Exception:
            pass
        if self.sample_nvidia:
            q = _nvidia_smi_query()
            if q:
                self.result.nvidia_peak_used_mib = max(
                    self.result.nvidia_peak_used_mib, q.get("memory_used_mib", 0.0))
        self.result.samples += 1

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample_once()

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "ResourceMonitor":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._t0 = time.perf_counter()
        self._sample_once()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample_once()
        self.result.seconds = time.perf_counter() - self._t0

    def save(self, path: Path | str, **extra) -> Path:
        from .hashing import atomic_write_json
        payload = self.result.as_dict()
        payload.update(extra)
        payload["host"] = host_report()
        return atomic_write_json(path, payload)


def write_hardware_report(path: Path | str) -> dict:
    from .hashing import atomic_write_json
    info = host_report()
    atomic_write_json(path, info)
    return info


class Timer:
    """Minimal wall-clock timer usable as a context manager."""

    def __init__(self, label: str = ""):
        self.label = label
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0

    def __repr__(self) -> str:  # pragma: no cover
        return f"Timer({self.label!r}, {self.seconds:.3f}s)"
