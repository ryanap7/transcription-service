"""
Custom exceptions for Audio Transcription System
"""


class AudioTranscriptionError(Exception):
    """Base exception for audio transcription errors"""
    pass


class AudioFileError(AudioTranscriptionError):
    """Exception raised for audio file related errors"""
    pass


class AudioFormatError(AudioFileError):
    """Exception raised for unsupported audio formats"""
    pass


class AudioSizeError(AudioFileError):
    """Exception raised when audio file is too large"""
    pass


class ModelLoadError(AudioTranscriptionError):
    """Exception raised when model loading fails"""
    pass


class TranscriptionError(AudioTranscriptionError):
    """Exception raised during transcription process"""
    pass


class DiarizationError(AudioTranscriptionError):
    """Exception raised during speaker diarization"""
    pass


class SummarizationError(AudioTranscriptionError):
    """Exception raised during AI summarization"""
    pass


class ConfigurationError(AudioTranscriptionError):
    """Exception raised for configuration errors"""
    pass