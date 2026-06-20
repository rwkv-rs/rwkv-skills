import logging
from threading import Lock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from API_servers.router import openai_router, state_router, v1_router
from infer.metrics import GpuSampler


def create_app(engine, password=None):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.engine = engine
    app.state.password = password
    app.state.dialogue_idx_lock = Lock()
    app.state.dialogue_idx_counters = {}

    # Background GPU sampler, scoped to this backend's CUDA_VISIBLE_DEVICES.
    gpu_sampler = GpuSampler()
    app.state.gpu_sampler = gpu_sampler
    if getattr(engine, "metrics", None) is not None:
        engine.metrics.gpu_sampler = gpu_sampler

    app.include_router(v1_router)
    app.include_router(state_router)
    app.include_router(openai_router)

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok",
            "model": _model_name(engine),
            "engine": "rwkv-lightning",
        }

    @app.get("/v1/backpressure")
    @app.get("/v1/batch-metrics")
    async def batch_metrics():
        pending = _pending_snapshot(engine)
        metrics = _metrics_snapshot(engine)
        gpu = gpu_sampler.snapshot()
        max_batch_size = pending["max_batch_size"]
        reserved = pending["prefill_reserved_bsz"]
        active_decode = metrics["active_decode_bsz"]
        return {
            "status": "ok",
            "model_name": _model_name(engine),
            "max_batch_size": max_batch_size,
            "batch_collect_ms": 0,
            "pending": pending,
            "throughput": metrics["throughput"],
            "gpu": gpu,
            "occupancy": {
                "active_decode_bsz": active_decode,
                "prefill_reserved_bsz": reserved,
                "max_batch_size": max_batch_size,
                "batch_fill_pct": _pct(reserved, max_batch_size),
                "active_decode_fill_pct": _pct(active_decode, max_batch_size),
            },
            "totals": metrics["totals"],
            "backend_live": {
                "engine": "rwkv-lightning",
                "prefill_reserved_bsz": reserved,
                "prefill_queue": pending["service_queue"],
            },
            "recent_batches": metrics["recent_batches"],
        }

    @app.on_event("startup")
    async def _start_gpu_sampler():
        gpu_sampler.start()

    @app.on_event("shutdown")
    async def _stop_gpu_sampler():
        gpu_sampler.stop()

    @app.on_event("startup")
    async def log_registered_routes():
        logger = logging.getLogger("uvicorn.error")
        logger.info("Registered FastAPI routes:")
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue
            methods = ",".join(sorted(route.methods - {"HEAD", "OPTIONS"}))
            logger.info("  %-20s %s", methods, route.path)

    return app


def _model_name(engine):
    return str(getattr(getattr(engine, "args", object()), "MODEL_NAME", "rwkv7")).split("/")[-1]


def _pct(value, total):
    if not total or total <= 0:
        return 0.0
    return round(100.0 * float(value) / float(total), 1)


def _metrics_snapshot(engine):
    metrics = getattr(engine, "metrics", None)
    if metrics is None:
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
    return metrics.snapshot()


def _pending_snapshot(engine):
    prefill_queue = getattr(engine, "_prefill_queue", ())
    try:
        service_queue = len(prefill_queue)
    except TypeError:
        service_queue = 0
    model = getattr(engine, "model", object())
    max_batch_size = int(
        getattr(
            model,
            "max_prefill_bsz_limit",
            getattr(model, "max_prefill_bsz", 0),
        )
        or 0
    )
    return {
        "pending_queue": service_queue,
        "service_queue": service_queue,
        "engine_inbox": 0,
        "active_records": 0,
        "scheduler_waiting": service_queue,
        "scheduler_running": 0,
        "prefill_reserved_bsz": int(getattr(engine, "_prefill_reserved_bsz", 0) or 0),
        "max_batch_size": max_batch_size,
    }
