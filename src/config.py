"""
Configuration settings for the Audio Processing API
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    APP_NAME: str = "Audio Transcription API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # Keep 1 for GPU models
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # File Upload
    MAX_FILE_SIZE_MB: int = 200
    ALLOWED_EXTENSIONS: List[str] = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    
    # Audio Processing
    TARGET_SAMPLE_RATE: int = 16000
    NORMALIZE_DB: float = -20.0
    
    # Whisper Settings
    WHISPER_MODEL: str = "large-v3"  # tiny, base, small, medium, large, large-v3
    WHISPER_LANGUAGE: str = "id"  # Indonesian
    WHISPER_DEVICE: str = "cuda"  # cuda or cpu
    
    # Pyannote Settings
    HUGGINGFACE_TOKEN: str = ""  # Required: Get from https://huggingface.co/settings/tokens
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
    
    # Claude API
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS: int = 1500
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/api.log"
    
    # Performance
    MERGE_GAP_THRESHOLD: float = 1.0  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()