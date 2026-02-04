"""
Speaker diarization using Pyannote
"""

import logging
from typing import List, Dict, Optional
import torch
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)


class PyannoteDiarizer:
    """Speaker diarization with Pyannote"""
    
    def __init__(self, hf_token: str, model_name: str = "pyannote/speaker-diarization-3.1"):
        """
        Initialize diarizer
        
        Args:
            hf_token: HuggingFace access token
            model_name: Pyannote model name
        """
        if not hf_token or "your_" in hf_token.lower():
            raise ValueError(
                "HuggingFace token required!\n"
                "1. Get token: https://huggingface.co/settings/tokens\n"
                "2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
        
        self.hf_token = hf_token
        self.model_name = model_name
        self.pipeline = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Pyannote on {self._device}...")
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load Pyannote pipeline"""
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token
            )
            
            if self._device == "cuda":
                self.pipeline.to(torch.device("cuda"))
                logger.info("✅ Pyannote loaded on GPU")
            else:
                logger.info("✅ Pyannote loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load Pyannote: {e}")
            raise ValueError(f"Pyannote initialization failed: {e}")
    
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
        """
        Perform speaker diarization
        
        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers (None for auto-detect)
            
        Returns:
            List of diarization segments with speaker labels
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        try:
            logger.info(f"Running diarization (num_speakers={num_speakers})...")
            
            # Run diarization
            if num_speakers:
                result = self.pipeline(audio_path, num_speakers=num_speakers)
            else:
                result = self.pipeline(audio_path)
            
            # Extract segments
            segments = []
            diarization = result.speaker_diarization if hasattr(result, 'speaker_diarization') else result
            
            for segment, track, label in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': float(segment.start),
                    'end': float(segment.end),
                    'speaker': label,
                    'duration': float(segment.end - segment.start)
                })
            
            # Rename speakers to SPEAKER_1, SPEAKER_2, etc.
            segments = self._rename_speakers(segments)
            
            unique_speakers = len(set(s['speaker'] for s in segments))
            logger.info(f"✅ Diarization complete: {unique_speakers} speakers, {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise RuntimeError(f"Speaker diarization failed: {e}")
    
    @staticmethod
    def _rename_speakers(segments: List[Dict]) -> List[Dict]:
        """
        Rename speaker labels to SPEAKER_1, SPEAKER_2, etc.
        
        Args:
            segments: List of segments with original speaker labels
            
        Returns:
            Segments with renamed speakers
        """
        unique_speakers = sorted(set(s['speaker'] for s in segments))
        speaker_map = {old: f'SPEAKER_{i+1}' for i, old in enumerate(unique_speakers)}
        
        for seg in segments:
            seg['speaker'] = speaker_map[seg['speaker']]
        
        return segments