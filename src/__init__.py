"""
Audio Transcription System
"""

__version__ = "1.0.0"
__author__ = "Audio Transcription Team"

from .core import Config
from .utils import AudioTranscriptionPipeline

__all__ = ['AudioTranscriptionPipeline', 'Config']