"""
Speech transcription using OpenAI Whisper - Optimized in-memory processing
"""
from typing import List, Dict, Optional, Tuple
import whisper
import torch
import numpy as np

from src.core.config import Config
from src.core.exceptions import TranscriptionError, ModelLoadError


class WhisperTranscriber:
    """Speech-to-text transcription using Whisper - Optimized"""

    def __init__(self, model_size: Optional[str] = None):
        """
        Initialize transcriber

        Args:
            model_size: Whisper model size (uses config if not provided)

        Raises:
            ModelLoadError: If model loading fails
        """
        self.model_size = model_size or Config.WHISPER_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model"""
        try:
            print(f"Loading Whisper model: {self.model_size}...")
            self.model = whisper.load_model(self.model_size)

            # Enable optimizations
            if torch.cuda.is_available():
                print("Using GPU for transcription")
                # Enable TF32 for faster processing on Ampere GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            print("Whisper model loaded successfully")
        except Exception as e:
            raise ModelLoadError(f"Failed to load Whisper model: {str(e)}")

    def transcribe_tensor(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = 'transcribe'
    ) -> Dict[str, any]:
        """
        Transcribe audio from tensor - Faster in-memory processing

        Args:
            audio_tensor: Audio tensor (1D or 2D)
            sample_rate: Sample rate (should be 16000 for Whisper)
            language: Language code (uses config if not provided)
            task: 'transcribe' or 'translate'

        Returns:
            Transcription result with segments

        Raises:
            TranscriptionError: If transcription fails
        """
        try:
            language = language or Config.LANGUAGE

            print(f"Transcribing audio (language: {language})...")

            # Convert tensor to numpy array for Whisper
            if isinstance(audio_tensor, torch.Tensor):
                # Remove channel dimension if present
                if audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor.squeeze(0)
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor

            # Ensure it's float32 and in the right range
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)

            # Whisper expects audio in [-1, 1] range
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

            # Transcribe with optimizations - ENHANCED FOR SPEED
            result = self.model.transcribe(
                audio_np,
                language=language,
                task=task,
                verbose=False,
                fp16=torch.cuda.is_available(),  # Use FP16 on GPU for speed
                condition_on_previous_text=False,  # Faster processing
                temperature=0.0,  # Greedy decoding for speed
                beam_size=1,  # Greedy search (faster than beam search)
                best_of=1,  # No sampling (deterministic and faster)
                compression_ratio_threshold=2.4,  # Skip segments with poor compression
                no_speech_threshold=0.6,  # Skip silent segments
                logprob_threshold=-1.0,  # Skip low-confidence segments
            )

            print(f"Transcription complete: {len(result['segments'])} segments")

            return result

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)}")

    def transcribe_with_speakers(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int,
        diarization_segments: List[Dict],
        language: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Transcribe audio tensor with speaker diarization - Optimized

        Args:
            audio_tensor: Audio tensor
            sample_rate: Sample rate
            diarization_segments: Speaker diarization segments
            language: Language code

        Returns:
            List of segments with speaker labels and transcriptions

        Raises:
            TranscriptionError: If transcription fails
        """
        try:
            # Get full transcription
            result = self.transcribe_tensor(audio_tensor, sample_rate, language)
            whisper_segments = result['segments']

            print(f"Aligning {len(whisper_segments)} Whisper segments with "
                  f"{len(diarization_segments)} diarization segments...")

            # Align Whisper segments with diarization
            aligned = self._align_segments(whisper_segments, diarization_segments)

            print(f"Alignment complete: {len(aligned)} final segments")

            return aligned

        except Exception as e:
            raise TranscriptionError(f"Transcription with speakers failed: {str(e)}")

    @staticmethod
    def _align_segments(
        whisper_segments: List[Dict],
        diarization_segments: List[Dict]
    ) -> List[Dict[str, any]]:
        """
        Align Whisper transcription segments with diarization segments - Optimized

        Args:
            whisper_segments: Segments from Whisper
            diarization_segments: Segments from diarization

        Returns:
            Aligned segments with speaker and text
        """
        aligned = []

        # Pre-calculate for faster lookup
        total_segments = len(diarization_segments)

        for i, dia_seg in enumerate(diarization_segments):
            # Show progress every 20 segments
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{total_segments}", end='\r')

            text_parts = []
            dia_start = dia_seg['start']
            dia_end = dia_seg['end']

            # Find overlapping Whisper segments
            for whisper_seg in whisper_segments:
                whisper_start = whisper_seg['start']
                whisper_end = whisper_seg['end']

                # Quick overlap check
                if whisper_end < dia_start or whisper_start > dia_end:
                    continue

                # Calculate overlap
                overlap_start = max(whisper_start, dia_start)
                overlap_end = min(whisper_end, dia_end)
                overlap = overlap_end - overlap_start

                # Check if there's significant overlap
                whisper_duration = whisper_end - whisper_start
                if overlap > whisper_duration * Config.OVERLAP_THRESHOLD:
                    text_parts.append(whisper_seg['text'].strip())

            # Combine text
            combined_text = ' '.join(text_parts).strip()

            if combined_text:
                aligned.append({
                    'speaker': dia_seg['speaker'],
                    'start': dia_seg['start'],
                    'end': dia_seg['end'],
                    'duration': dia_seg['duration'],
                    'text': combined_text
                })

        print()  # New line after progress
        return aligned

    @staticmethod
    def merge_consecutive_segments(
        segments: List[Dict],
        gap_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Merge consecutive segments from same speaker - Optimized

        Args:
            segments: List of segments
            gap_threshold: Max gap between segments to merge (uses config if not provided)

        Returns:
            Merged segments
        """
        if not segments:
            return []

        gap_threshold = gap_threshold or Config.MERGE_GAP_THRESHOLD

        merged = []
        current = segments[0].copy()

        for seg in segments[1:]:
            gap = seg['start'] - current['end']

            # Merge if same speaker and gap is small
            if seg['speaker'] == current['speaker'] and gap < gap_threshold:
                current['end'] = seg['end']
                current['duration'] = current['end'] - current['start']
                current['text'] += ' ' + seg['text']
            else:
                merged.append(current)
                current = seg.copy()

        merged.append(current)

        print(f"Merged {len(segments)} segments into {len(merged)} segments")

        return merged