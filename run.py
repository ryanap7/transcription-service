#!/usr/bin/env python3
"""
Run Audio Transcription API Server
"""
import sys

print("=" * 80)
print("SIMTELPAS AI - Starting up...")
print("=" * 80)

# Preload models
print("\n[1/3] Loading configuration...")
from src.core.config import Config
print(f"✓ Model: {Config.WHISPER_MODEL}")
print(f"✓ Language: {Config.LANGUAGE}")

print("\n[2/3] Initializing AI models (this may take a while)...")
try:
    from src.utils.pipeline import AudioTranscriptionPipeline
    pipeline = AudioTranscriptionPipeline()
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"✗ Failed to load models: {e}")
    print("\nPlease check your configuration and try again.")
    sys.exit(1)

print("\n[3/3] Starting API server...")
from src.api.api import main

if __name__ == '__main__':
    main()