from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import os
import logging
from pathlib import Path

from src.pipeline import AudioPipeline
from src.config import get_settings
from src.models import (
    AudioProcessResponse,
    HealthCheckResponse,
    ErrorResponse
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Create logs directory
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# APP INITIALIZATION
# ============================================================================

settings = get_settings()

app = FastAPI(
    title="Audio Transcription & Diarization API",
    description="""
    🎙️ Professional audio processing API with:
    - 👥 Speaker Diarization (Pyannote)
    - 📝 Speech-to-Text Transcription (Whisper)
    - 💡 AI-Powered Summaries (Claude)
    
    Supported formats: WAV, MP3, M4A, FLAC, OGG
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Health check endpoints"},
        {"name": "transcription", "description": "Audio transcription endpoints"}
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AudioPipeline] = None

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global pipeline
    logger.info("="*80)
    logger.info("🚀 STARTING API SERVER")
    logger.info("="*80)
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"Whisper Model: {settings.WHISPER_MODEL}")
    logger.info(f"Device: {settings.WHISPER_DEVICE}")
    
    try:
        pipeline = AudioPipeline(settings)
        logger.info("="*80)
        logger.info("✅ API SERVER READY")
        logger.info("="*80)
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down API server...")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cleanup_temp_file(filepath: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"🗑️  Cleaned up: {filepath}")
    except Exception as e:
        logger.warning(f"⚠️  Failed to cleanup {filepath}: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get(
    "/",
    response_model=HealthCheckResponse,
    tags=["health"],
    summary="Root endpoint"
)
async def root():
    """Root endpoint - API information"""
    return {
        "status": "online",
        "message": "Audio Transcription & Diarization API",
        "version": "1.0.0"
    }

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["health"],
    summary="Health check"
)
async def health_check():
    """
    Health check endpoint
    Returns API status and model loading status
    """
    return {
        "status": "healthy",
        "message": "All systems operational",
        "version": "1.0.0",
        "models_loaded": pipeline is not None
    }

@app.post(
    "/api/v1/transcribe",
    response_model=AudioProcessResponse,
    responses={
        200: {
            "description": "Successfully processed audio",
            "model": AudioProcessResponse
        },
        400: {
            "description": "Bad request - invalid file or parameters",
            "model": ErrorResponse
        },
        413: {
            "description": "File too large",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    },
    tags=["transcription"],
    summary="Transcribe audio with speaker diarization",
    description="""
    ## Upload and process audio file
    
    This endpoint performs:
    1. Audio normalization
    2. Speaker diarization (identify who speaks when)
    3. Speech-to-text transcription
    4. Transcript formatting and statistics
    5. Optional AI summarization
    
    ### Supported Formats
    - WAV, MP3, M4A, FLAC, OGG
    - Max file size: 200MB
    - Recommended: 16kHz mono audio
    
    ### Parameters
    - **file**: Audio file (required)
    - **num_speakers**: Expected number of speakers (optional, auto-detect if not provided)
    - **include_summary**: Generate AI summary (optional, requires Anthropic API key)
    
    ### Response
    Returns JSON with:
    - Full transcript with timestamps
    - Individual segments with speaker labels
    - Statistics (duration, word count, speaker breakdown)
    - Optional AI summary
    """
)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ...,
        description="Audio file to transcribe (WAV, MP3, M4A, FLAC, OGG)"
    ),
    num_speakers: Optional[int] = Query(
        None,
        ge=1,
        le=10,
        description="Expected number of speakers (auto-detect if not provided)"
    ),
    include_summary: bool = Query(
        False,
        description="Generate AI summary using Claude (requires API key)"
    )
):
    """
    Process audio file: normalization → diarization → transcription → formatting
    
    **Example Usage:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/transcribe" \\
         -F "file=@audio.wav" \\
         -F "num_speakers=2" \\
         -F "include_summary=true"
    ```
    """
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable - models not loaded"
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. "
                   f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Create temporary file
    temp_file = None
    try:
        # Read file content
        content = await file.read()
        file_size_mb = len(content) / 1024 / 1024
        
        # Validate file size
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f}MB). "
                       f"Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            temp_file = tmp.name
        
        logger.info(
            f"📁 Processing: {file.filename} "
            f"({file_size_mb:.2f}MB, {num_speakers or 'auto'} speakers)"
        )
        
        # Process audio through pipeline
        result = pipeline.process(
            audio_path=temp_file,
            num_speakers=num_speakers,
            include_summary=include_summary
        )
        
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_temp_file, temp_file)
        if result.get('normalized_path'):
            background_tasks.add_task(cleanup_temp_file, result['normalized_path'])
        
        logger.info(f"✅ Completed: {file.filename}")
        
        # Return standardized response
        return AudioProcessResponse(
            success=True,
            message="Audio processed successfully",
            data={
                "audio_info": result['audio_info'],
                "transcript": result['transcript'],
                "segments": result['segments'],
                "statistics": result['statistics'],
                "summary": result.get('summary')
            }
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        if temp_file:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        raise
    
    except Exception as e:
        logger.error(
            f"❌ Error processing {file.filename}: {str(e)}",
            exc_info=True
        )
        
        # Cleanup on error
        if temp_file:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

# ============================================================================
# GLOBAL ERROR HANDLER
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc)
        }
    )

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )