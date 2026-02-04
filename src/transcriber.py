"""
Audio transcription using Whisper
"""

import logging
from typing import List, Dict
import whisper

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Whisper-based audio transcription"""
    
    def __init__(self, model_size: str = "large-v3", language: str = "id", device: str = "cuda"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v3)
            language: Language code (e.g., 'id' for Indonesian, 'en' for English)
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.model = None
        
        logger.info(f"Loading Whisper model: {model_size}")
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"✅ Whisper {self.model_size} loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise ValueError(f"Whisper initialization failed: {e}")
    
    def transcribe_with_segments(
        self,
        audio_path: str,
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """
        Transcribe audio with diarization segments
        
        Args:
            audio_path: Path to audio file
            diarization_segments: Speaker diarization segments
            
        Returns:
            List of transcribed segments with speaker labels
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        try:
            logger.info(f"Transcribing {len(diarization_segments)} segments...")
            
            # Run Whisper transcription
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                verbose=False,
                word_timestamps=False
            )
            
            whisper_segments = result['segments']
            logger.info(f"Whisper generated {len(whisper_segments)} segments")
            
            # Match Whisper segments with diarization
            transcribed = self._match_segments(diarization_segments, whisper_segments)
            
            # Filter empty segments
            transcribed = [seg for seg in transcribed if seg['text'].strip()]
            
            logger.info(f"✅ Transcription complete: {len(transcribed)} segments")
            return transcribed
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Audio transcription failed: {e}")
    
    @staticmethod
    def _match_segments(
        diarization_segments: List[Dict],
        whisper_segments: List[Dict]
    ) -> List[Dict]:
        """
        Match Whisper transcription with diarization segments
        
        Args:
            diarization_segments: Speaker diarization segments
            whisper_segments: Whisper transcription segments
            
        Returns:
            Merged segments with speaker labels and transcription
        """
        transcribed = []
        total = len(diarization_segments)
        
        for i, dia_seg in enumerate(diarization_segments):
            # Progress logging
            if (i + 1) % 10 == 0 or (i + 1) == total:
                progress = (i + 1) / total * 100
                logger.debug(f"Progress: {progress:.1f}% ({i+1}/{total})")
            
            text_parts = []
            
            # Find overlapping Whisper segments
            for w_seg in whisper_segments:
                overlap_start = max(w_seg['start'], dia_seg['start'])
                overlap_end = min(w_seg['end'], dia_seg['end'])
                overlap = overlap_end - overlap_start
                
                # If overlap is significant (>30% of Whisper segment)
                if overlap > 0:
                    w_duration = w_seg['end'] - w_seg['start']
                    if overlap > w_duration * 0.3:
                        text_parts.append(w_seg['text'].strip())
            
            # Combine text
            combined_text = ' '.join(text_parts)
            
            if combined_text.strip():
                transcribed.append({
                    'speaker': dia_seg['speaker'],
                    'start': dia_seg['start'],
                    'end': dia_seg['end'],
                    'duration': dia_seg['duration'],
                    'text': combined_text
                })
        
        return transcribed