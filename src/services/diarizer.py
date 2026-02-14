"""
Speaker diarization using pyannote.audio - In-memory processing
"""
from typing import List, Dict, Optional, BinaryIO
import torch
from pyannote.audio import Pipeline
import io
import soundfile as sf

from src.core.config import Config
from src.core.exceptions import DiarizationError, ModelLoadError


class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio"""

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize diarizer

        Args:
            hf_token: HuggingFace token (uses config if not provided)

        Raises:
            ModelLoadError: If model loading fails
        """
        self.hf_token = hf_token or Config.HUGGINGFACE_TOKEN

        if not self.hf_token:
            raise ModelLoadError(
                "HuggingFace token required for speaker diarization. "
                "Get token from: https://huggingface.co/settings/tokens "
                "and accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load pyannote pipeline"""
        try:
            print("Loading speaker diarization model...")

            # Enable TF32 for better performance on GPU
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Suppress warnings
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)

            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                print("Using GPU for diarization")
            else:
                print("Using CPU for diarization")

        except Exception as e:
            raise ModelLoadError(f"Failed to load diarization model: {str(e)}")

    def diarize(
        self,
        audio_input: BinaryIO,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Perform speaker diarization - Now accepts BytesIO instead of file path

        Args:
            audio_input: BytesIO object containing WAV audio data
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            List of diarization segments with speaker labels

        Raises:
            DiarizationError: If diarization fails
        """
        try:
            print(f"Performing speaker diarization...")

            # Suppress warnings during diarization
            import warnings
            warnings.filterwarnings('ignore')

            # Ensure we're at the beginning of the stream
            if hasattr(audio_input, 'seek'):
                audio_input.seek(0)

            # Load audio from BytesIO using soundfile
            waveform, sample_rate = sf.read(audio_input)

            # Convert to torch tensor and ensure correct shape
            if len(waveform.shape) == 1:
                # Mono audio
                waveform = torch.from_numpy(waveform).float().unsqueeze(0)
            else:
                # Stereo - take first channel
                waveform = torch.from_numpy(waveform[:, 0]).float().unsqueeze(0)

            # Prepare parameters
            params = {}
            if num_speakers:
                params['num_speakers'] = num_speakers
            else:
                if min_speakers:
                    params['min_speakers'] = min_speakers
                if max_speakers:
                    params['max_speakers'] = max_speakers

            # Run diarization with proper format
            audio_dict = {
                'waveform': waveform,
                'sample_rate': sample_rate,
                'uri': 'audio'
            }

            result = self.pipeline(audio_dict, **params)

            # Extract speaker_diarization from DiarizeOutput (pyannote 3.1+)
            # The result is a DiarizeOutput object with speaker_diarization attribute
            if hasattr(result, 'speaker_diarization'):
                diarization = result.speaker_diarization
            elif hasattr(result, 'itertracks'):
                # Fallback for older versions
                diarization = result
            else:
                raise DiarizationError(f"Unknown result format: {type(result)}")

            # Now iterate through the annotation object
            segments = []
            for segment, track, label in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'duration': segment.end - segment.start,
                    'speaker': label
                })

            # Rename speakers to standard format
            segments = self._rename_speakers(segments)

            num_speakers_found = len(set(s['speaker'] for s in segments))
            print(f"Found {num_speakers_found} speakers in {len(segments)} segments")

            return segments

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise DiarizationError(f"Diarization failed: {str(e)}")

    @staticmethod
    def _rename_speakers(segments: List[Dict]) -> List[Dict]:
        """
        Rename speakers to standard format (SPEAKER_1, SPEAKER_2, etc.)

        Args:
            segments: List of segments with speaker labels

        Returns:
            Segments with renamed speakers
        """
        # Get unique speakers sorted by first appearance
        unique_speakers = []
        for seg in segments:
            if seg['speaker'] not in unique_speakers:
                unique_speakers.append(seg['speaker'])

        # Create mapping
        speaker_map = {
            old: f'SPEAKER_{i+1}'
            for i, old in enumerate(unique_speakers)
        }

        # Rename
        for seg in segments:
            seg['speaker'] = speaker_map[seg['speaker']]

        return segments

    def get_speaker_statistics(self, segments: List[Dict]) -> Dict[str, Dict]:
        """
        Get statistics for each speaker

        Args:
            segments: List of diarization segments

        Returns:
            Dictionary with speaker statistics
        """
        stats = {}

        for seg in segments:
            speaker = seg['speaker']

            if speaker not in stats:
                stats[speaker] = {
                    'total_duration': 0,
                    'num_segments': 0,
                    'segments': []
                }

            stats[speaker]['total_duration'] += seg['duration']
            stats[speaker]['num_segments'] += 1
            stats[speaker]['segments'].append(seg)

        return stats