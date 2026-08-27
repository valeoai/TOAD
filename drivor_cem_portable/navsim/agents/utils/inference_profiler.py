"""Lightweight per-section inference timer for measuring MPC/scorer overhead.

Each model wraps its forward into named sections (e.g. ``vanilla``, ``cem``,
``drivor_rescore``); the timer accumulates GPU-synchronized wall-clock time
per section and prints a rolling per-sample summary every ``log_every`` calls.

The intended use is to read the ``avg ms/sample`` rows of the report and
compare a ``use_cem=False`` run against a ``use_cem=True`` run on the same
data: the difference between the ``total`` rows is the extra time introduced
by the scorer + CEM/MPC refinement.

Multi-GPU (DDP): each rank has its own ``_PROFILER`` instance (it's a
module-level global, so each Python process gets a fresh one). By default
only rank 0 prints the rolling report, to keep stdout clean across the 8
processes. Set ``INFER_PROFILE_ALL_RANKS=1`` to make every rank print.
"""

import os
import time
from contextlib import contextmanager
from collections import OrderedDict

import torch


def _detect_rank():
    """Best-effort rank lookup that works with torchrun / Lightning DDP."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_rank()
        except Exception:
            pass
    for env_key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(env_key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    return 0


class InferenceTimer:
    def __init__(self, name: str, log_every: int = 50, enabled: bool = True):
        self.name = name
        self.log_every = int(os.environ.get("INFER_PROFILE_LOG_EVERY", log_every))
        env_enabled = os.environ.get("INFER_PROFILE", "1") not in ("0", "false", "False")
        self.enabled = enabled and env_enabled
        self._all_ranks = os.environ.get("INFER_PROFILE_ALL_RANKS", "0") not in ("0", "false", "False")
        self._total_time = OrderedDict()
        self._total_samples = OrderedDict()
        self._call_count = 0

    @contextmanager
    def section(self, key: str, batch_size: int = 1):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self._total_time[key] = self._total_time.get(key, 0.0) + dt
            self._total_samples[key] = self._total_samples.get(key, 0) + int(batch_size)

    def increment(self):
        if not self.enabled:
            return
        self._call_count += 1
        if self.log_every > 0 and self._call_count % self.log_every == 0:
            self.report()

    def report(self):
        if not self.enabled or not self._total_time:
            return
        rank = _detect_rank()
        if rank != 0 and not self._all_ranks:
            return
        lines = [f"[InferenceTimer:{self.name}][rank={rank}] calls={self._call_count}"]
        for key, total in self._total_time.items():
            samples = max(self._total_samples.get(key, 0), 1)
            avg_ms = (total / samples) * 1000.0
            lines.append(
                f"  {key:>20s}: avg={avg_ms:8.2f} ms/sample  "
                f"n_samples={samples}  total={total:.2f}s"
            )
        print("\n".join(lines), flush=True)
