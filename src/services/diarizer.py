"""
Speaker diarization - SILENT version (no prints during spinner)
"""
from typing import List, Dict, Optional, BinaryIO
import torch
from pyannote.audio import Pipeline
import soundfile as sf

from src.core.config import Config
from src.core.exceptions import DiarizationError, ModelLoadError


class SpeakerDiarizer:
    """Speaker diarization - silent processing"""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or Config.HUGGINGFACE_TOKEN
        if not self.hf_token:
            raise ModelLoadError("HuggingFace token required")
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load model - only print during init"""
        try:
            print("Loading speaker diarization model...", flush=True)

            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)

            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            )

            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                print("Using GPU for diarization", flush=True)
            else:
                print("Using CPU for diarization", flush=True)

        except Exception as e:
            raise ModelLoadError(f"Failed to load diarization model: {str(e)}")

    def diarize(
        self,
        audio_input: BinaryIO,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """Diarize - SILENT (no prints)"""
        try:
            # NO PRINT - spinner is active!

            import warnings
            warnings.filterwarnings('ignore')

            if hasattr(audio_input, 'seek'):
                audio_input.seek(0)

            waveform, sample_rate = sf.read(audio_input)

            if len(waveform.shape) == 1:
                waveform = torch.from_numpy(waveform).float().unsqueeze(0)
            else:
                waveform = torch.from_numpy(waveform[:, 0]).float().unsqueeze(0)

            params = {}
            if num_speakers:
                params['num_speakers'] = num_speakers
            else:
                if min_speakers:
                    params['min_speakers'] = min_speakers
                if max_speakers:
                    params['max_speakers'] = max_speakers

            audio_dict = {
                'waveform': waveform,
                'sample_rate': sample_rate,
                'uri': 'audio'
            }

            result = self.pipeline(audio_dict, **params)

            if hasattr(result, 'speaker_diarization'):
                diarization = result.speaker_diarization
            elif hasattr(result, 'itertracks'):
                diarization = result
            else:
                raise DiarizationError(f"Unknown result format: {type(result)}")

            segments = []
            for segment, track, label in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'duration': segment.end - segment.start,
                    'speaker': label
                })

            segments = self._rename_speakers(segments)

            # NO PRINT - pipeline will show count
            return segments

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise DiarizationError(f"Diarization failed: {str(e)}")

    @staticmethod
    def _rename_speakers(segments: List[Dict]) -> List[Dict]:
        """Rename speakers - silent"""
        unique_speakers = []
        for seg in segments:
            if seg['speaker'] not in unique_speakers:
                unique_speakers.append(seg['speaker'])

        speaker_map = {
            old: f'SPEAKER_{i+1}'
            for i, old in enumerate(unique_speakers)
        }

        for seg in segments:
            seg['speaker'] = speaker_map[seg['speaker']]

        return segments

    def get_speaker_statistics(self, segments: List[Dict]) -> Dict[str, Dict]:
        """Get speaker stats"""
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