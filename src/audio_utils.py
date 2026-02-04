"""
Audio utility functions for processing
"""

import os
import logging
from typing import Dict
from pathlib import Path
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class AudioUtils:
    """Audio processing utilities"""
    
    @staticmethod
    def get_audio_info(audio_path: str) -> Dict:
        """
        Get audio file information
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio metadata
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            return {
                'duration_sec': len(audio) / 1000.0,
                'duration_min': len(audio) / 60000.0,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels
            }
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            raise ValueError(f"Invalid audio file: {e}")
    
    @staticmethod
    def normalize_audio(audio_path: str, target_dbfs: float = -20.0) -> str:
        """
        Normalize audio volume
        
        Args:
            audio_path: Input audio file path
            target_dbfs: Target volume in dBFS
            
        Returns:
            Path to normalized audio file
        """
        try:
            logger.info(f"Normalizing audio: {audio_path}")
            
            audio = AudioSegment.from_file(audio_path)
            change_in_dbfs = target_dbfs - audio.dBFS
            normalized = audio.apply_gain(change_in_dbfs)
            
            # Create output path
            output_path = str(Path(audio_path).with_suffix('')) + '_normalized.wav'
            
            # Export as WAV for processing
            normalized.export(output_path, format='wav')
            
            logger.info(f"Audio normalized: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to normalize audio: {e}")
            raise ValueError(f"Audio normalization failed: {e}")
    
    @staticmethod
    def validate_audio_file(filepath: str) -> bool:
        """
        Validate audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            True if valid, raises exception otherwise
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        if os.path.getsize(filepath) == 0:
            raise ValueError("Audio file is empty")
        
        # Try to load the file
        try:
            AudioSegment.from_file(filepath)
            return True
        except Exception as e:
            raise ValueError(f"Invalid audio file: {e}")