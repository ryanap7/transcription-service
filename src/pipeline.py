"""
Main audio processing pipeline
"""

import logging
from typing import Dict, Optional
from pathlib import Path

from .config import Settings
from .audio_utils import AudioUtils
from .diarizer import PyannoteDiarizer
from .transcriber import WhisperTranscriber
from .formatter import TranscriptFormatter
from .summarizer import AISummarizer

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Complete audio processing pipeline:
    Audio -> Normalize -> Diarize -> Transcribe -> Format -> Summarize
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize pipeline with all components
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        logger.info("🚀 Initializing Audio Pipeline...")
        
        # Initialize components
        self.utils = AudioUtils()
        
        self.diarizer = PyannoteDiarizer(
            hf_token=settings.HUGGINGFACE_TOKEN,
            model_name=settings.DIARIZATION_MODEL
        )
        
        self.transcriber = WhisperTranscriber(
            model_size=settings.WHISPER_MODEL,
            language=settings.WHISPER_LANGUAGE,
            device=settings.WHISPER_DEVICE
        )
        
        self.formatter = TranscriptFormatter()
        
        self.summarizer = AISummarizer(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.CLAUDE_MODEL
        )
        
        logger.info("✅ Pipeline initialized successfully")
    
    def process(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        include_summary: bool = False
    ) -> Dict:
        """
        Process audio file through complete pipeline
        
        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers (None for auto-detect)
            include_summary: Whether to generate AI summary
            
        Returns:
            Dictionary with all results:
            - audio_info: Audio metadata
            - segments: Transcribed segments
            - transcript: Formatted transcript
            - statistics: Statistics
            - summary: AI summary (if requested)
            - normalized_path: Path to normalized audio (for cleanup)
        """
        try:
            logger.info("="*80)
            logger.info("🎙️  AUDIO PROCESSING PIPELINE")
            logger.info("="*80)
            
            # 1. Validate audio
            logger.info(f"📁 Input file: {Path(audio_path).name}")
            self.utils.validate_audio_file(audio_path)
            
            # 2. Get audio info
            audio_info = self.utils.get_audio_info(audio_path)
            logger.info(
                f"⏱️  Duration: {audio_info['duration_min']:.2f} min, "
                f"Sample Rate: {audio_info['sample_rate']} Hz, "
                f"Channels: {audio_info['channels']}"
            )
            
            # 3. Normalize audio
            logger.info("🔊 Normalizing audio...")
            normalized_path = self.utils.normalize_audio(
                audio_path,
                target_dbfs=self.settings.NORMALIZE_DB
            )
            
            # 4. Speaker diarization
            logger.info("👥 Running speaker diarization...")
            diarization_segments = self.diarizer.diarize(
                normalized_path,
                num_speakers=num_speakers
            )
            
            # 5. Transcription
            logger.info("📝 Transcribing audio...")
            transcribed_segments = self.transcriber.transcribe_with_segments(
                normalized_path,
                diarization_segments
            )
            
            # 6. Format transcript
            logger.info("📋 Formatting transcript...")
            merged_segments = self.formatter.merge_consecutive(
                transcribed_segments,
                gap_threshold=self.settings.MERGE_GAP_THRESHOLD
            )
            
            transcript = self.formatter.format_text(merged_segments)
            statistics = self.formatter.get_stats(merged_segments)
            
            # 7. Generate summary (optional)
            summary = None
            if include_summary:
                logger.info("💡 Generating AI summary...")
                summary = self.summarizer.create_summary(transcript, statistics)
            
            # Log results
            logger.info("="*80)
            logger.info("✅ PROCESSING COMPLETE")
            logger.info("="*80)
            logger.info(f"📊 Duration: {statistics['total_duration']/60:.1f} minutes")
            logger.info(f"📊 Total Words: {statistics['total_words']:,}")
            logger.info(f"📊 Speakers: {statistics['num_speakers']}")
            logger.info(f"📊 Segments: {len(merged_segments)}")
            
            for speaker, data in sorted(statistics['speakers'].items()):
                logger.info(
                    f"   {speaker}: {data['duration']/60:.1f} min "
                    f"({data['percentage']:.1f}%), {data['words']} words"
                )
            
            return {
                'audio_info': audio_info,
                'segments': merged_segments,
                'transcript': transcript,
                'statistics': statistics,
                'summary': summary,
                'normalized_path': normalized_path  # For cleanup
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise