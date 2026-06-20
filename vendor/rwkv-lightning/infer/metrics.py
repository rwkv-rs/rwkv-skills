"""Live observability for the rwkv-lightning inference engine.

Two independent, side-band components (failures here must never break inference):

* ``MetricsCollector`` accumulates throughput counters and a ring of per-batch
  records. It distinguishes ``decode_token_slots`` (the work the GPU actually
  did -- the full ``batch_size`` is forwarded every decode step even after some
  sequences have stopped) from ``output_tokens`` (tokens delivered to users).
* ``GpuSampler`` polls NVML on a background thread, honouring
  ``CUDA_VISIBLE_DEVICES`` so a single backend only reports its own GPU.

The ``/v1/batch-metrics`` endpoint reads ``MetricsCollector.snapshot()`` and
``GpuSampler.snapshot()`` to expose tok/s, GPU utilisation and batch fill.
"""

import os
import threading
import time
from collections import deque


def _now():
    # monotonic clock shared by MetricsCollector and GpuSampler so per-batch
    # time windows line up with GPU samples.
    return time.monotonic()


class _DecodeSession:
    """Context manager wrapping one decode loop (one batch).

    Usage::

        with engine.metrics.decode_session(batch_size, prefill_tokens=...) as sess:
            for _ in range(max_length):
                out = model.forward_batch(...)
                sess.step(active_slots)   # active_slots = sequences still emitting
    """

    def __init__(self, collector, bsz, prefill_tokens):
        self._c = collector
        self.bsz = max(1, int(bsz))
        self.prefill_tokens = int(prefill_tokens or 0)
        self.decode_token_slots = 0
        self.output_tokens = 0
        self.steps = 0
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = _now()
        self._c._decode_started(self.bsz)
        return self

    def step(self, active_slots):
        # Every step forwards the whole batch, so GPU work == bsz slots; only the
        # still-active sequences contribute user-visible output tokens.
        active = max(0, int(active_slots))
        self.decode_token_slots += self.bsz
        self.output_tokens += active
        self.steps += 1
        self._c._record_step(self.bsz, active)

    def __exit__(self, exc_type, exc, tb):
        self._c._decode_finished(
            bsz=self.bsz,
            prefill_tokens=self.prefill_tokens,
            decode_token_slots=self.decode_token_slots,
            output_tokens=self.output_tokens,
            elapsed_s=max(_now() - self.t0, 1e-9),
            t0=self.t0,
            t1=_now(),
            success=exc_type is None,
        )
        return False  # never suppress exceptions

    def finish(self, exc=None):
        """Close the session from a manual try/finally (async generators).

        Encapsulates the with-protocol so call sites never pass ``type(None)``
        for the success case -- ``exc is None`` is the only "succeeded" signal.
        """
        if exc is None:
            self.__exit__(None, None, None)
        else:
            self.__exit__(type(exc), exc, exc.__traceback__)


class MetricsCollector:
    def __init__(self, window_seconds=10.0, recent_capacity=64):
        self._lock = threading.Lock()
        self.window_seconds = float(window_seconds)
        # Set by create_app once the sampler exists; used to annotate per-batch
        # records with the GPU utilisation observed during that batch.
        self.gpu_sampler = None

        # Lifetime counters.
        self.decode_token_slots_total = 0
        self.output_tokens_total = 0
        self.prefill_tokens_total = 0
        self.decode_steps_total = 0
        self.total_batches = 0
        self.completed_requests = 0
        self.error_requests = 0
        self.failed_batches = 0

        # Busy time used for lifetime averages.
        self.decode_busy_s = 0.0
        self.prefill_busy_s = 0.0

        # Live gauge: slots currently being decoded (sums across concurrent batches).
        self.active_decode_bsz = 0

        # Peaks taken from clean per-batch measurements (not the noisy window).
        self.decode_token_slots_s_peak = 0.0
        self.output_tok_s_peak = 0.0

        # Rolling windows for the "is it busy right now" view.
        self._decode_window = deque()   # (ts, slots, output)
        self._prefill_window = deque()  # (ts, tokens, elapsed_s)

        self._recent = deque(maxlen=int(recent_capacity))

    # -- decode hooks (called from _DecodeSession) --------------------------

    def decode_session(self, bsz, prefill_tokens=0):
        return _DecodeSession(self, bsz, prefill_tokens)

    def _decode_started(self, bsz):
        with self._lock:
            self.active_decode_bsz += bsz

    def _record_step(self, bsz, active):
        ts = _now()
        with self._lock:
            self.decode_token_slots_total += bsz
            self.output_tokens_total += active
            self.decode_steps_total += 1
            self._decode_window.append((ts, bsz, active))
            self._trim_decode(ts)

    def _decode_finished(self, bsz, prefill_tokens, decode_token_slots,
                         output_tokens, elapsed_s, t0, t1, success):
        slots_s = decode_token_slots / elapsed_s if elapsed_s > 0 else 0.0
        out_s = output_tokens / elapsed_s if elapsed_s > 0 else 0.0
        gpu = None
        sampler = self.gpu_sampler
        if sampler is not None:
            try:
                gpu = sampler.window_stats(t0, t1)
            except Exception:
                gpu = None
        with self._lock:
            self.active_decode_bsz = max(0, self.active_decode_bsz - bsz)
            self.total_batches += 1
            self.decode_busy_s += elapsed_s
            if not success:
                self.failed_batches += 1
            self.decode_token_slots_s_peak = max(self.decode_token_slots_s_peak, slots_s)
            self.output_tok_s_peak = max(self.output_tok_s_peak, out_s)
            self._recent.append({
                "bsz": int(bsz),
                "prefill_tokens": int(prefill_tokens),
                "decode_token_slots": int(decode_token_slots),
                "output_tokens": int(output_tokens),
                "elapsed_ms": round(elapsed_s * 1000.0, 2),
                "decode_token_slots_s": round(slots_s, 2),
                "output_tok_s": round(out_s, 2),
                "success": bool(success),
                "gpu": gpu,
            })

    # -- prefill / request hooks -------------------------------------------

    def record_prefill(self, tokens, elapsed_s):
        tokens = int(tokens or 0)
        if tokens <= 0:
            return
        ts = _now()
        elapsed_s = max(0.0, float(elapsed_s))
        with self._lock:
            self.prefill_tokens_total += tokens
            self.prefill_busy_s += elapsed_s
            # Keep the real duration: windowed tok/s divides by sum(elapsed),
            # not by the timestamp span (which collapses to ~0 right after a
            # multi-second prefill and spikes the rate).
            self._prefill_window.append((ts, tokens, elapsed_s))
            self._trim_prefill(ts)

    def record_request(self, success):
        with self._lock:
            if success:
                self.completed_requests += 1
            else:
                self.error_requests += 1

    # -- window helpers (assume lock held) ---------------------------------

    def _trim_decode(self, now):
        cutoff = now - self.window_seconds
        win = self._decode_window
        while win and win[0][0] < cutoff:
            win.popleft()

    def _trim_prefill(self, now):
        cutoff = now - self.window_seconds
        win = self._prefill_window
        while win and win[0][0] < cutoff:
            win.popleft()

    # -- read --------------------------------------------------------------

    def snapshot(self):
        now = _now()
        with self._lock:
            self._trim_decode(now)
            self._trim_prefill(now)

            d_slots = sum(s for _, s, _ in self._decode_window)
            d_out = sum(o for _, _, o in self._decode_window)
            d_span = max(now - self._decode_window[0][0], 1e-6) if self._decode_window else 0.0
            p_tokens = sum(t for _, t, _ in self._prefill_window)
            p_elapsed = sum(e for _, _, e in self._prefill_window)

            decode_slots_s = d_slots / d_span if d_span > 0 else 0.0
            output_tok_s = d_out / d_span if d_span > 0 else 0.0
            prefill_tok_s = p_tokens / p_elapsed if p_elapsed > 0 else 0.0

            slots_s_avg = (self.decode_token_slots_total / self.decode_busy_s
                           if self.decode_busy_s > 0 else 0.0)
            out_s_avg = (self.output_tokens_total / self.decode_busy_s
                         if self.decode_busy_s > 0 else 0.0)
            prefill_s_avg = (self.prefill_tokens_total / self.prefill_busy_s
                             if self.prefill_busy_s > 0 else 0.0)

            return {
                "throughput": {
                    "decode_token_slots_s": round(decode_slots_s, 2),
                    "output_tok_s": round(output_tok_s, 2),
                    "prefill_tok_s": round(prefill_tok_s, 2),
                    "decode_token_slots_s_avg": round(slots_s_avg, 2),
                    "output_tok_s_avg": round(out_s_avg, 2),
                    "prefill_tok_s_avg": round(prefill_s_avg, 2),
                    "decode_token_slots_s_peak": round(self.decode_token_slots_s_peak, 2),
                    "output_tok_s_peak": round(self.output_tok_s_peak, 2),
                    "decode_token_slots_total": self.decode_token_slots_total,
                    "output_tokens_total": self.output_tokens_total,
                    "prefill_tokens_total": self.prefill_tokens_total,
                    "decode_steps_total": self.decode_steps_total,
                },
                "active_decode_bsz": self.active_decode_bsz,
                "totals": {
                    "total_batches": self.total_batches,
                    "total_requests": self.completed_requests + self.error_requests,
                    "completed_requests": self.completed_requests,
                    "error_requests": self.error_requests,
                    "failed_batches": self.failed_batches,
                },
                "recent_batches": list(self._recent),
            }


class _NullSession:
    def __enter__(self):
        return self

    def step(self, active_slots):
        pass

    def __exit__(self, exc_type, exc, tb):
        return False

    def finish(self, exc=None):
        return False


class _NullMetrics:
    """No-op stand-in so mixins used without an engine (e.g. tests) stay safe."""

    gpu_sampler = None

    def decode_session(self, bsz, prefill_tokens=0):
        return _NullSession()

    def record_prefill(self, tokens, elapsed_s):
        pass

    def record_request(self, success):
        pass

    def snapshot(self):
        return {
            "throughput": {},
            "active_decode_bsz": 0,
            "totals": {
                "total_batches": 0,
                "total_requests": 0,
                "completed_requests": 0,
                "error_requests": 0,
                "failed_batches": 0,
            },
            "recent_batches": [],
        }


NULL_METRICS = _NullMetrics()


def get_metrics(engine):
    """Return the engine's collector, or a null collector if absent."""
    return getattr(engine, "metrics", None) or NULL_METRICS


class GpuSampler:
    """Background NVML poller scoped to ``CUDA_VISIBLE_DEVICES``."""

    def __init__(self, interval_s=1.0, history=120):
        self.interval_s = float(interval_s)
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._nvml = None
        self._handles = []          # list of (logical, physical, handle)
        self._available = False
        self._error = None
        self._visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._latest = []           # list of per-gpu dicts
        self._ewma = {}             # physical_index -> smoothed sm util
        self._peak = {}             # physical_index -> peak sm util
        self._samples = deque(maxlen=int(history))  # (ts, max_sm, total_mem_used_mb)
        self._init_nvml()

    # -- setup -------------------------------------------------------------

    def _resolve_visible(self, count):
        """Map CUDA_VISIBLE_DEVICES to (logical_pos, physical_index) pairs."""
        raw = self._visible
        if raw is None:
            return [(i, i) for i in range(count)]
        tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]
        if not tokens:
            return []
        pairs = []
        for logical, tok in enumerate(tokens):
            if tok == "-1":
                break  # CUDA treats -1 (and everything after) as "no device"
            try:
                phys = int(tok)
            except ValueError:
                # UUID form (e.g. "GPU-xxxx"): resolve via NVML if possible.
                phys = self._index_from_uuid(tok)
                if phys is None:
                    continue
            if 0 <= phys < count:
                pairs.append((logical, phys))
        return pairs

    def _index_from_uuid(self, uuid):
        try:
            handle = self._nvml.nvmlDeviceGetHandleByUUID(uuid.encode())
            return int(self._nvml.nvmlDeviceGetIndex(handle))
        except Exception:
            return None

    def _init_nvml(self):
        try:
            import pynvml  # noqa: PLC0415 -- optional, deprecation warning is harmless
            pynvml.nvmlInit()
            self._nvml = pynvml
            count = pynvml.nvmlDeviceGetCount()
            for logical, phys in self._resolve_visible(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(phys)
                self._handles.append((logical, phys, handle))
            self._available = len(self._handles) > 0
            if not self._available and self._error is None:
                self._error = "no visible CUDA devices"
        except Exception as exc:  # pynvml missing, no driver, etc.
            self._available = False
            self._error = repr(exc)

    # -- background loop ---------------------------------------------------

    def start(self):
        if not self._available or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="gpu-sampler", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _run(self):
        nv = self._nvml
        while not self._stop.is_set():
            latest = []
            max_sm = 0
            total_mem = 0.0
            for logical, phys, handle in self._handles:
                try:
                    util = nv.nvmlDeviceGetUtilizationRates(handle)
                    mem = nv.nvmlDeviceGetMemoryInfo(handle)
                    sm = int(util.gpu)
                    used_mb = mem.used / (1024.0 * 1024.0)
                    prev = self._ewma.get(phys)
                    ewma = sm if prev is None else (0.3 * sm + 0.7 * prev)
                    self._ewma[phys] = ewma
                    self._peak[phys] = max(self._peak.get(phys, 0), sm)
                    latest.append({
                        "index": phys,
                        "logical_index": logical,
                        "name": self._device_name(handle),
                        "sm_util_pct": sm,
                        "sm_util_avg": round(ewma, 1),
                        "sm_util_peak": self._peak[phys],
                        "mem_util_pct": int(util.memory),
                        "mem_used_mb": round(used_mb, 1),
                        "mem_free_mb": round(mem.free / (1024.0 * 1024.0), 1),
                        "mem_total_mb": round(mem.total / (1024.0 * 1024.0), 1),
                        "power_w": self._power(handle),
                        "temp_c": self._temp(handle),
                    })
                    max_sm = max(max_sm, sm)
                    total_mem += used_mb
                except Exception as exc:
                    latest.append({"index": phys, "error": repr(exc)})
            with self._lock:
                self._latest = latest
                self._samples.append((_now(), max_sm, total_mem))
            self._stop.wait(self.interval_s)

    def _device_name(self, handle):
        try:
            name = self._nvml.nvmlDeviceGetName(handle)
            return name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            return None

    def _power(self, handle):
        try:
            return round(self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1)
        except Exception:
            return None

    def _temp(self, handle):
        try:
            return int(self._nvml.nvmlDeviceGetTemperature(
                handle, self._nvml.NVML_TEMPERATURE_GPU))
        except Exception:
            return None

    # -- read --------------------------------------------------------------

    def window_stats(self, t0, t1):
        """Aggregate SM util / memory observed in [t0, t1] (for one batch)."""
        with self._lock:
            pts = [(sm, mem) for ts, sm, mem in self._samples if t0 <= ts <= t1]
            if not pts and self._samples:
                pts = [(self._samples[-1][1], self._samples[-1][2])]
            if not pts:
                return None
            sms = [sm for sm, _ in pts]
            mems = [mem for _, mem in pts]
            return {
                "sm_util_avg": round(sum(sms) / len(sms), 1),
                "sm_util_peak": max(sms),
                "mem_used_mb": round(sum(mems) / len(mems), 1),
            }

    def snapshot(self):
        with self._lock:
            return {
                "available": self._available,
                "error": self._error,
                "visible_devices": self._visible,
                "gpus": list(self._latest),
            }
