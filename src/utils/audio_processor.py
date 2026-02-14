"""
Audio processing utilities - Pure in-memory processing
"""
import io
from pathlib import Path
from typing import Dict, BinaryIO, Union, Tuple

from pydub import AudioSegment
import numpy as np
import torch
import torchaudio

from src.core.config import Config
from src.core.exceptions import AudioFileError, AudioFormatError, AudioSizeError


class AudioProcessor:
    """Process audio files without saving - Pure in-memory"""

    @staticmethod
    def validate_audio(file_path_or_bytes: Union[str, Path, bytes, BinaryIO]) -> Dict[str, any]:
        """
        Validate audio file/stream

        Args:
            file_path_or_bytes: Audio file path or bytes stream

        Returns:
            Dictionary containing audio information

        Raises:
            AudioFileError: If audio cannot be loaded
            AudioFormatError: If format is not supported
            AudioSizeError: If file is too large
        """
        try:
            # Load audio
            if isinstance(file_path_or_bytes, (str, Path)):
                file_path = Path(file_path_or_bytes)

                # Check file extension
                if file_path.suffix.lower() not in Config.SUPPORTED_FORMATS:
                    raise AudioFormatError(
                        f"Unsupported format: {file_path.suffix}. "
                        f"Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}"
                    )

                # Check file size
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > Config.MAX_AUDIO_SIZE_MB:
                    raise AudioSizeError(
                        f"File too large: {file_size_mb:.1f}MB. "
                        f"Maximum size: {Config.MAX_AUDIO_SIZE_MB}MB"
                    )

                audio = AudioSegment.from_file(str(file_path))
            else:
                # Handle bytes or file-like object
                audio = AudioSegment.from_file(file_path_or_bytes)

            # Get audio information
            info = {
                'duration_seconds': len(audio) / 1000.0,
                'duration_minutes': len(audio) / 60000.0,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'format': 'audio',
                'bit_depth': audio.sample_width * 8
            }

            return info

        except AudioFormatError:
            raise
        except AudioSizeError:
            raise
        except Exception as e:
            raise AudioFileError(f"Failed to load audio: {str(e)}")

    @staticmethod
    def prepare_for_processing(
        file_path_or_bytes: Union[str, Path, bytes, BinaryIO],
        normalize: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Prepare audio for processing - Pure in-memory, returns torch tensor

        Args:
            file_path_or_bytes: Audio file path or bytes stream
            normalize: Whether to normalize audio

        Returns:
            Tuple of (audio_tensor, sample_rate)

        Raises:
            AudioFileError: If processing fails
        """
        try:
            # Load audio with better error handling
            if isinstance(file_path_or_bytes, (str, Path)):
                audio = AudioSegment.from_file(str(file_path_or_bytes))
            else:
                # For BytesIO/stream, pass directly to AudioSegment
                if hasattr(file_path_or_bytes, 'seek'):
                    file_path_or_bytes.seek(0)
                audio = AudioSegment.from_file(file_path_or_bytes)

            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Resample to target rate
            if audio.frame_rate != Config.TARGET_SAMPLE_RATE:
                audio = audio.set_frame_rate(Config.TARGET_SAMPLE_RATE)

            # Normalize volume
            if normalize:
                change_in_dbfs = Config.NORMALIZE_DB - audio.dBFS
                audio = audio.apply_gain(change_in_dbfs)

            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())

            # Normalize to [-1, 1] range
            if audio.sample_width == 2:  # 16-bit
                samples = samples.astype(np.float32) / 32768.0
            elif audio.sample_width == 4:  # 32-bit
                samples = samples.astype(np.float32) / 2147483648.0
            else:
                samples = samples.astype(np.float32)

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(samples).float()

            # Add channel dimension if needed (for torchaudio compatibility)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            return audio_tensor, Config.TARGET_SAMPLE_RATE

        except Exception as e:
            raise AudioFileError(
                f"Failed to prepare audio. "
                f"Please ensure the file is a valid audio format (mp3, wav, m4a, flac, ogg). "
                f"Details: {str(e)}"
            )

    @staticmethod
    def prepare_wav_bytes(
        file_path_or_bytes: Union[str, Path, bytes, BinaryIO],
        normalize: bool = True
    ) -> io.BytesIO:
        """
        Prepare audio as WAV bytes in-memory for pyannote

        Args:
            file_path_or_bytes: Audio file path or bytes stream
            normalize: Whether to normalize audio

        Returns:
            BytesIO object containing WAV data

        Raises:
            AudioFileError: If processing fails
        """
        try:
            # Load audio
            if isinstance(file_path_or_bytes, (str, Path)):
                audio = AudioSegment.from_file(str(file_path_or_bytes))
            else:
                if hasattr(file_path_or_bytes, 'seek'):
                    file_path_or_bytes.seek(0)
                audio = AudioSegment.from_file(file_path_or_bytes)

            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Resample to target rate
            if audio.frame_rate != Config.TARGET_SAMPLE_RATE:
                audio = audio.set_frame_rate(Config.TARGET_SAMPLE_RATE)

            # Normalize volume
            if normalize:
                change_in_dbfs = Config.NORMALIZE_DB - audio.dBFS
                audio = audio.apply_gain(change_in_dbfs)

            # Export to BytesIO as WAV
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format='wav')
            wav_buffer.seek(0)

            return wav_buffer

        except Exception as e:
            raise AudioFileError(f"Failed to prepare WAV bytes: {str(e)}")

    @staticmethod
    def get_audio_stats(file_path_or_bytes: Union[str, Path, bytes, BinaryIO]) -> Dict[str, any]:
        """
        Get detailed audio statistics

        Args:
            file_path_or_bytes: Audio file path or bytes stream

        Returns:
            Dictionary with audio statistics
        """
        try:
            if isinstance(file_path_or_bytes, (str, Path)):
                audio = AudioSegment.from_file(str(file_path_or_bytes))
            else:
                if hasattr(file_path_or_bytes, 'seek'):
                    file_path_or_bytes.seek(0)
                audio = AudioSegment.from_file(file_path_or_bytes)

            return {
                'duration_seconds': len(audio) / 1000.0,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'bit_depth': audio.sample_width * 8,
                'dbfs': audio.dBFS,
                'max_dbfs': audio.max_dBFS,
                'rms': audio.rms
            }
        except Exception as e:
            raise AudioFileError(f"Failed to get audio stats: {str(e)}")