"""
Main audio transcription pipeline with REAL-TIME PROGRESS
Pure in-memory processing with detailed timing
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


class ProgressTimer:
    """Live progress timer - updates every 100ms"""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = time.time()
        self.running = True
        self.thread = None

    def _update(self):
        """Update timer display in real-time"""
        while self.running:
            elapsed = time.time() - self.start_time
            print(f"\r⏱️  {self.step_name}: {elapsed:.1f}s", end='', flush=True)
            time.sleep(0.1)  # Update 10 times per second

    def start(self):
        """Start the live timer"""
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop timer and show final time"""
        self.running = False
        if self.thread:
            self.thread.join()
        elapsed = time.time() - self.start_time
        print(f"\r✓ {self.step_name}: {elapsed:.2f}s                ", flush=True)
        return elapsed


class AudioTranscriptionPipeline:
    """Complete audio transcription pipeline"""

    def __init__(
        self,
        whisper_model: Optional[str] = None,
        language: Optional[str] = None,
        enable_summary: bool = True
    ):
        """Initialize pipeline with progress tracking"""
        is_valid, errors = Config.validate()
        if not is_valid:
            raise ConfigurationError(
                f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        print("Initializing Audio Transcription Pipeline...", flush=True)
        print("=" * 80, flush=True)

        self.audio_processor = AudioProcessor()
        self.diarizer = SpeakerDiarizer()
        self.transcriber = WhisperTranscriber(whisper_model)
        self.summarizer = AISummarizer() if enable_summary else None
        self.language = language or Config.LANGUAGE

        print("=" * 80, flush=True)
        print("Pipeline initialized successfully\n", flush=True)

    def process(
        self,
        audio_input: Union[bytes, BinaryIO],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        include_summary: bool = True
    ) -> Dict[str, any]:
        """Process audio with REAL-TIME progress tracking"""

        timings = {}
        total_start = time.time()

        try:
            print("\n" + "=" * 80, flush=True)
            print("STARTING AUDIO TRANSCRIPTION", flush=True)
            print("=" * 80, flush=True)

            # Step 1: Validate audio WITH LIVE TIMER
            timer = ProgressTimer("[1/5] Validating audio")
            timer.start()
            audio_info = self.audio_processor.validate_audio(audio_input)
            timings['validation'] = timer.stop()
            print(f"    Duration: {audio_info['duration_minutes']:.2f} minutes\n", flush=True)

            # Step 2: Prepare audio WITH LIVE TIMER
            timer = ProgressTimer("[2/5] Preparing audio (in-memory)")
            timer.start()

            audio_tensor, sample_rate = self.audio_processor.prepare_for_processing(audio_input)
            wav_bytes = self.audio_processor.prepare_wav_bytes(audio_input)

            timings['preparation'] = timer.stop()
            print(f"    Tensor: {audio_tensor.shape}\n", flush=True)

            # Step 3: Speaker diarization WITH LIVE TIMER
            timer = ProgressTimer("[3/5] Speaker diarization")
            timer.start()

            diarization_segments = self.diarizer.diarize(
                wav_bytes,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )

            timings['diarization'] = timer.stop()
            num_spk = len(set(s['speaker'] for s in diarization_segments))
            print(f"    Found {num_spk} speakers, {len(diarization_segments)} segments\n", flush=True)

            # Step 4: Transcription WITH LIVE TIMER
            timer = ProgressTimer("[4/5] Transcribing audio")
            timer.start()

            transcribed_segments = self.transcriber.transcribe_with_speakers(
                audio_tensor,
                sample_rate,
                diarization_segments,
                language=self.language
            )

            merged_segments = self.transcriber.merge_consecutive_segments(transcribed_segments)

            timings['transcription'] = timer.stop()
            print(f"    Segments: {len(merged_segments)} final\n", flush=True)

            # Step 5: AI Summary WITH LIVE TIMER
            summary = None
            if include_summary and self.summarizer and self.summarizer.is_available():
                timer = ProgressTimer("[5/5] Generating AI summary")
                timer.start()

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

                timings['summarization'] = timer.stop()
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
        """Display timing breakdown"""
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