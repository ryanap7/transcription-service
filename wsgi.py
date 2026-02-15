#!/usr/bin/env python3
"""
WSGI Entry Point - Preload AI models with post-fork CUDA initialization
Handles CUDA fork safety for Gunicorn workers
"""
import sys
import os

# Set multiprocessing start method to 'spawn' for CUDA compatibility
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

print("=" * 80)
print("SIMTELPAS AI - Preloading Models...")
print("=" * 80)

# Load configuration
print("\n[1/3] Loading configuration...")
from src.core.config import Config
print(f"✓ Model: {Config.WHISPER_MODEL}")
print(f"✓ Language: {Config.LANGUAGE}")

# Import modules but DON'T initialize CUDA yet
print("\n[2/3] Preparing pipeline (CUDA will init after fork)...")
from src.utils.pipeline import AudioTranscriptionPipeline

# Create pipeline instance but models will load lazily in workers
global_pipeline = AudioTranscriptionPipeline()

print("✓ Pipeline structure ready")

# Inject pipeline into Flask app
print("\n[3/3] Loading Flask application...")
import src.api.api as api_module
api_module.pipeline = global_pipeline
print("✓ Pipeline injected into Flask app")

# Import Flask app
from src.api.api import app

print("\n" + "=" * 80)
print("✓ Ready to accept requests!")
print("✓ CUDA models will initialize in workers after fork")
print("=" * 80)
print()

# Gunicorn post_fork hook to reinitialize CUDA in workers
def post_fork(server, worker):
    """
    Called after worker process is forked
    Reinitialize CUDA-dependent components here
    """
    import torch
    if torch.cuda.is_available():
        # Clear CUDA cache
        torch.cuda.empty_cache()
        print(f"Worker {worker.pid}: CUDA reinitialized")