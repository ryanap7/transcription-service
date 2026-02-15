"""
Gunicorn Configuration - Post-fork CUDA initialization
"""
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"

# Worker processes
workers = 2
worker_class = "sync"
worker_tmp_dir = "/dev/shm"

# Timeouts
timeout = 900
graceful_timeout = 30
keepalive = 5

# Worker lifecycle
max_requests = 200
max_requests_jitter = 20

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'


def post_worker_init(worker):
    """
    Called just after a worker has been forked
    Initialize pipeline with CUDA here (safe after fork)
    """
    import src.api.api as api_module
    from src.utils.pipeline import AudioTranscriptionPipeline

    print(f"\n{'=' * 80}")
    print(f"[Worker {worker.pid}] Initializing pipeline with CUDA...")
    print(f"{'=' * 80}\n")

    try:
        # Create pipeline - CUDA will initialize here, AFTER fork
        pipeline = AudioTranscriptionPipeline()

        # Inject into api module
        api_module.pipeline = pipeline

        print(f"\n{'=' * 80}")
        print(f"[Worker {worker.pid}] ✓ Pipeline ready with CUDA")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"[Worker {worker.pid}] ✗ Failed to initialize pipeline: {e}")
        print(f"{'=' * 80}\n")
        import traceback
        traceback.print_exc()
        raise


def worker_exit(server, worker):
    """Called just after a worker has been exited"""
    print(f"[Worker {worker.pid}] Shutting down...")


def on_exit(server):
    """Called just before the master process exits"""
    print("\nGunicorn master shutting down...")