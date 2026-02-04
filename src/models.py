"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# REQUEST MODELS
# ============================================================================

class TranscribeRequest(BaseModel):
    """Request model for transcription (multipart form)"""
    num_speakers: Optional[int] = Field(None, ge=1, le=10, description="Number of speakers (auto-detect if None)")
    include_summary: bool = Field(False, description="Generate AI summary")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class AudioInfo(BaseModel):
    """Audio file information"""
    duration_sec: float = Field(..., description="Duration in seconds")
    duration_min: float = Field(..., description="Duration in minutes")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    channels: int = Field(..., description="Number of channels")


class SpeakerStats(BaseModel):
    """Statistics for individual speaker"""
    duration: float = Field(..., description="Total speaking time in seconds")
    turns: int = Field(..., description="Number of speaking turns")
    words: int = Field(..., description="Total word count")
    percentage: float = Field(..., description="Percentage of total speaking time")


class Statistics(BaseModel):
    """Overall statistics"""
    total_duration: float = Field(..., description="Total audio duration in seconds")
    total_words: int = Field(..., description="Total word count")
    num_speakers: int = Field(..., description="Number of detected speakers")
    speakers: Dict[str, SpeakerStats] = Field(..., description="Per-speaker statistics")


class Segment(BaseModel):
    """Transcript segment"""
    speaker: str = Field(..., description="Speaker label (e.g., SPEAKER_1)")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")
    text: str = Field(..., description="Transcribed text")


class AudioProcessData(BaseModel):
    """Main data response"""
    audio_info: AudioInfo
    transcript: str = Field(..., description="Full formatted transcript")
    segments: List[Segment] = Field(..., description="Individual segments")
    statistics: Statistics
    summary: Optional[str] = Field(None, description="AI-generated summary (if requested)")


class AudioProcessResponse(BaseModel):
    """Standard API response for transcription"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[AudioProcessData] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Audio processed successfully",
                "data": {
                    "audio_info": {
                        "duration_sec": 120.5,
                        "duration_min": 2.0,
                        "sample_rate": 16000,
                        "channels": 1
                    },
                    "transcript": "SPEAKER_1 [00:00 - 00:05]: Hello...",
                    "segments": [
                        {
                            "speaker": "SPEAKER_1",
                            "start": 0.0,
                            "end": 5.2,
                            "duration": 5.2,
                            "text": "Hello, how are you?"
                        }
                    ],
                    "statistics": {
                        "total_duration": 120.5,
                        "total_words": 250,
                        "num_speakers": 2,
                        "speakers": {
                            "SPEAKER_1": {
                                "duration": 60.0,
                                "turns": 5,
                                "words": 120,
                                "percentage": 49.8
                            }
                        }
                    },
                    "summary": "AI-generated summary here..."
                },
                "timestamp": "2024-02-04T10:30:00"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    message: str = Field(..., description="Status message")
    version: Optional[str] = Field(None, description="API version")
    models_loaded: Optional[bool] = Field(None, description="Whether models are loaded")


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = Field(False, description="Always false for errors")
    message: str = Field(..., description="Error message")
    error: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Processing failed",
                "error": "File format not supported",
                "timestamp": "2024-02-04T10:30:00"
            }
        }