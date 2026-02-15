"""
Main audio transcription pipeline with CLEAN REAL-TIME PROGRESS
Uses spinner animation instead of timer to avoid text conflicts
"""
import time
import sys
from typing import Dict, Optional, Union, BinaryIO
from threading import Thread

from src.core.config import Config
from src.core.exceptions import *
from src.utils.audio_processor import AudioProcessor
from src.services.diarizer import SpeakerDiarizer
from src.services.transcriber import WhisperTranscriber
from src.services.summarizer import AISummarizer
from src.utils.formatter import TranscriptFormatter


class SpinnerProgress:
    """Clean spinner progress indicator"""

    SPINNERS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, message: str):
        self.message = message
        self.start_time = time.time()
        self.running = False
        self.thread = None
        self.idx = 0

    def _spin(self):
        """Animate spinner"""
        while self.running:
            elapsed = time.time() - self.start_time
            spinner = self.SPINNERS[self.idx % len(self.SPINNERS)]
            print(f"\r  {spinner} {self.message} ({elapsed:.0f}s)", end='', flush=True)
            self.idx += 1
            time.sleep(0.1)

    def start(self):
        """Start spinner"""
        print(f"\n{self.message}", flush=True)
        self.running = True
        self.thread = Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self, success=True):
        """Stop spinner and show result"""
        self.running = False
        if self.thread:
            self.thread.join()

        elapsed = time.time() - self.start_time
        symbol = '✓' if success else '✗'
        # Clear spinner line and print final result
        print(f"\r  {symbol} {self.message} - {elapsed:.2f}s" + " " * 20, flush=True)
        return elapsed


class AudioTranscriptionPipeline:
    """Complete audio transcription pipeline with clean progress"""

    def __init__(
        self,
        whisper_model: Optional[str] = None,
        language: Optional[str] = None,
        enable_summary: bool = True
    ):
        """Initialize pipeline"""
        is_valid, errors = Config.validate()
        if not is_valid:
            raise ConfigurationError(
                f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        print("\n" + "=" * 80, flush=True)
        print("Initializing Audio Transcription Pipeline", flush=True)
        print("=" * 80, flush=True)

        self.audio_processor = AudioProcessor()
        self.diarizer = SpeakerDiarizer()
        self.transcriber = WhisperTranscriber(whisper_model)
        self.summarizer = AISummarizer() if enable_summary else None
        self.language = language or Config.LANGUAGE

        print("=" * 80, flush=True)
        print("✓ Pipeline ready\n", flush=True)

    def process(
        self,
        audio_input: Union[bytes, BinaryIO],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        include_summary: bool = True
    ) -> Dict[str, any]:
        """Process audio with clean real-time progress"""

        timings = {}
        total_start = time.time()

        try:
            print("\n" + "=" * 80, flush=True)
            print("STARTING AUDIO TRANSCRIPTION", flush=True)
            print("=" * 80, flush=True)

            # Validate
            spinner = SpinnerProgress("[1/5] Validating audio")
            spinner.start()
            audio_info = self.audio_processor.validate_audio(audio_input)
            timings['validation'] = spinner.stop()
            print(f"      Duration: {audio_info['duration_minutes']:.2f} minutes", flush=True)

            # Prepare
            spinner = SpinnerProgress("[2/5] Preparing audio")
            spinner.start()
            audio_tensor, sample_rate = self.audio_processor.prepare_for_processing(audio_input)
            wav_bytes = self.audio_processor.prepare_wav_bytes(audio_input)
            timings['preparation'] = spinner.stop()
            print(f"      Tensor: {audio_tensor.shape}", flush=True)

            # Diarization
            spinner = SpinnerProgress("[3/5] Speaker diarization")
            spinner.start()
            diarization_segments = self.diarizer.diarize(
                wav_bytes,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            timings['diarization'] = spinner.stop()
            num_spk = len(set(s['speaker'] for s in diarization_segments))
            print(f"      Found {num_spk} speakers in {len(diarization_segments)} segments", flush=True)

            # Transcription
            spinner = SpinnerProgress("[4/5] Transcribing audio")
            spinner.start()
            transcribed_segments = self.transcriber.transcribe_with_speakers(
                audio_tensor,
                sample_rate,
                diarization_segments,
                language=self.language
            )
            merged_segments = self.transcriber.merge_consecutive_segments(transcribed_segments)
            timings['transcription'] = spinner.stop()
            print(f"      Generated {len(merged_segments)} final segments", flush=True)

            # Summary
            summary = None
            if include_summary and self.summarizer and self.summarizer.is_available():
                spinner = SpinnerProgress("[5/5] Generating AI summary")
                spinner.start()
                statistics = TranscriptFormatter.calculate_statistics(merged_segments)
                transcript_text = TranscriptFormatter.format_segments_to_text(
                    merged_segments,
                    include_timestamps=False
                )
                summary = self.summarizer.create_summary(
                    transcript_text,
                    statistics,
                    language=self.language
                )
                timings['summarization'] = spinner.stop()
            else:
                print("\n[5/5] Skipping AI summary", flush=True)
                timings['summarization'] = 0

            # Final stats
            statistics = TranscriptFormatter.calculate_statistics(merged_segments)
            timings['total'] = time.time() - total_start

            # Display summary
            self._display_timing_summary(timings)

            print("\n" + "=" * 80, flush=True)
            print("✓ PROCESSING COMPLETE", flush=True)
            print("=" * 80, flush=True)

            return {
                'success': True,
                'audio_info': audio_info,
                'segments': merged_segments,
                'statistics': statistics,
                'summary': summary,
                'timings': timings
            }

        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}", flush=True)
            raise

    def _display_timing_summary(self, timings: Dict[str, float]):
        """Display clean timing breakdown"""
        print("\n" + "=" * 80, flush=True)
        print("TIMING BREAKDOWN", flush=True)
        print("=" * 80, flush=True)

        steps = [
            ('validation', '1. Audio Validation'),
            ('preparation', '2. Audio Preparation'),
            ('diarization', '3. Speaker Diarization'),
            ('transcription', '4. Transcription'),
            ('summarization', '5. AI Summary')
        ]

        total = timings.get('total', 0)

        for key, label in steps:
            duration = timings.get(key, 0)
            percentage = (duration / total * 100) if total > 0 else 0
            bar_length = int(percentage / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"{label:.<30} {duration:>6.2f}s [{bar}] {percentage:>5.1f}%", flush=True)

        print("-" * 80, flush=True)
        print(f"{'TOTAL TIME':.<30} {total:>6.2f}s", flush=True)
        print("=" * 80, flush=True)