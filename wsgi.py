#!/usr/bin/env python3
"""
WSGI Entry Point - Lazy CUDA initialization for fork safety
Models are loaded AFTER worker fork to avoid CUDA context issues
"""
import sys
import os

print("=" * 80)
print("SIMTELPAS AI - Starting (Post-Fork CUDA Init)...")
print("=" * 80)

# Load configuration
print("\n[1/2] Loading configuration...")
from src.core.config import Config
print(f"✓ Model: {Config.WHISPER_MODEL}")
print(f"✓ Language: {Config.LANGUAGE}")

# Import Flask app (NO CUDA initialization yet)
print("\n[2/2] Loading Flask application...")

# Import api module
import src.api.api as api_module

# Pipeline will be None initially
api_module.pipeline = None

# Import Flask app
from src.api.api import app

print("\n" + "=" * 80)
print("✓ App loaded - Pipeline will init in workers after fork")
print("=" * 80)
print()

pipeline_initialized = False