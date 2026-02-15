#!/usr/bin/env python3
"""
WSGI Entry Point - Preload AI models before Gunicorn workers start
This ensures models are loaded ONCE and shared across all workers
"""
import sys

print("=" * 80)
print("SIMTELPAS AI - Preloading Models...")
print("=" * 80)

# Load configuration
print("\n[1/3] Loading configuration...")
from src.core.config import Config
print(f"✓ Model: {Config.WHISPER_MODEL}")
print(f"✓ Language: {Config.LANGUAGE}")

# Preload AI models
print("\n[2/3] Initializing AI models (this may take a while)...")
try:
    from src.utils.pipeline import AudioTranscriptionPipeline

    # Create global pipeline instance - will be shared by all workers
    global_pipeline = AudioTranscriptionPipeline()

    print("✓ All models loaded successfully")
except Exception as e:
    print(f"✗ Failed to load models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Inject pipeline into Flask app BEFORE importing app
print("\n[3/3] Loading Flask application...")

# Import api module and inject the preloaded pipeline
import src.api.api as api_module
api_module.pipeline = global_pipeline
print("✓ Pipeline injected into Flask app")

# Now import the app object for Gunicorn
from src.api.api import app

print("\n" + "=" * 80)
print("✓ Ready to accept requests!")
print("=" * 80)
print()