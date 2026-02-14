"""
Configuration module for Audio Transcription System - CLEAN VERSION
No file directories needed - pure in-memory processing
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration - Pure in-memory, no file operations"""

    # Base director
    BASE_DIR = Path(__file__).parent.parent.parent

    # Audio settings
    TARGET_SAMPLE_RATE = 16000
    NORMALIZE_DB = -20.0
    MAX_AUDIO_SIZE_MB = 500
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm'}

    # Model settings
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
    LANGUAGE = os.getenv('LANGUAGE', 'id')

    # API Keys
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

    # Processing settings
    MERGE_GAP_THRESHOLD = 1.0  # seconds
    OVERLAP_THRESHOLD = 0.3

    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        Validate configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check HuggingFace token
        if not cls.HUGGINGFACE_TOKEN:
            errors.append("HUGGINGFACE_TOKEN is required for speaker diarization")

        # Check Whisper model
        valid_models = {'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'}
        if cls.WHISPER_MODEL not in valid_models:
            errors.append(f"Invalid WHISPER_MODEL. Must be one of: {valid_models}")

        return len(errors) == 0, errors