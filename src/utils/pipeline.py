"""
Main audio transcription pipeline
Pure in-memory processing with detailed timing, NO file operations
"""
import time
from typing import Dict, Optional, Union, BinaryIO

from src.core.config import Config
from src.core.exceptions import *
from src.utils.audio_processor import AudioProcessor
from src.services.diarizer import SpeakerDiarizer
from src.services.transcriber import WhisperTranscriber
from src.services.summarizer import AISummarizer
from src.utils.formatter import TranscriptFormatter


class AudioTranscriptionPipeline:
    """
    Complete audio transcription pipeline
    """

    def __init__(
        self,
        whisper_model: Optional[str] = None,
        language: Optional[str] = None,
        enable_summary: bool = True
    ):
        """
        Initialize pipeline

        Args:
            whisper_model: Whisper model size
            language: Target language
            enable_summary: Whether to generate AI summary
        """
        # Validate configuration
        is_valid, errors = Config.validate()
        if not is_valid:
            raise ConfigurationError(
                f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Initialize components
        print("Initializing Audio Transcription Pipeline...")
        print("=" * 80)

        self.audio_processor = AudioProcessor()
        self.diarizer = SpeakerDiarizer()
        self.transcriber = WhisperTranscriber(whisper_model)
        self.summarizer = AISummarizer() if enable_summary else None

        self.language = language or Config.LANGUAGE

        print("=" * 80)
        print("Pipeline initialized successfully")
        print()

    def process(
        self,
        audio_input: Union[bytes, BinaryIO],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        include_summary: bool = True
    ) -> Dict[str, any]:
        """
        Process audio through complete pipeline - PURE IN-MEMORY with TIMING

        Args:
            audio_input: Audio bytes or file-like object
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            include_summary: Whether to generate AI summary

        Returns:
            Dictionary containing results with timing info (ready for API response)

        Raises:
            AudioTranscriptionError: If processing fails
        """
        # Track timing for each step
        timings = {}
        total_start = time.time()

        try:
            print("\n" + "=" * 80)
            print("STARTING AUDIO TRANSCRIPTION")
            print("=" * 80)

            # Step 1: Validate audio
            step_start = time.time()
            print("\n[1/5] Validating audio...")
            audio_info = self.audio_processor.validate_audio(audio_input)
            timings['validation'] = time.time() - step_start
            print(f"✓ Audio validated: {audio_info['duration_minutes']:.2f} minutes")
            print(f"⏱️  Time: {timings['validation']:.2f}s")

            # Step 2: Prepare audio for processing - IN MEMORY
            step_start = time.time()
            print("\n[2/5] Preparing audio (in-memory)...")

            # Prepare for Whisper (tensor)
            audio_tensor, sample_rate = self.audio_processor.prepare_for_processing(audio_input)
            print(f"✓ Audio tensor prepared: {audio_tensor.shape}")

            # Prepare for Diarization (WAV bytes)
            wav_bytes = self.audio_processor.prepare_wav_bytes(audio_input)
            print(f"✓ WAV bytes prepared for diarization")

            timings['preparation'] = time.time() - step_start
            print(f"⏱️  Time: {timings['preparation']:.2f}s")

            # Step 3: Speaker diarization - IN MEMORY
            step_start = time.time()
            print("\n[3/5] Performing speaker diarization...")
            diarization_segments = self.diarizer.diarize(
                wav_bytes,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            timings['diarization'] = time.time() - step_start
            print(f"✓ Diarization complete: {len(diarization_segments)} segments")
            print(f"⏱️  Time: {timings['diarization']:.2f}s")

            # Step 4: Transcription - IN MEMORY
            step_start = time.time()
            print("\n[4/5] Transcribing audio...")
            transcribed_segments = self.transcriber.transcribe_with_speakers(
                audio_tensor,
                sample_rate,
                diarization_segments,
                language=self.language
            )

            # Merge consecutive segments
            merged_segments = self.transcriber.merge_consecutive_segments(
                transcribed_segments
            )
            timings['transcription'] = time.time() - step_start
            print(f"✓ Transcription complete: {len(merged_segments)} final segments")
            print(f"⏱️  Time: {timings['transcription']:.2f}s")

            # Step 5: Generate summary
            summary = None
            if include_summary and self.summarizer and self.summarizer.is_available():
                step_start = time.time()
                print("\n[5/5] Generating concise AI summary...")
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
                timings['summarization'] = time.time() - step_start
                print("✓ Summary generated")
                print(f"⏱️  Time: {timings['summarization']:.2f}s")
            else:
                print("\n[5/5] Skipping AI summary")
                timings['summarization'] = 0

            # Calculate final statistics
            statistics = TranscriptFormatter.calculate_statistics(merged_segments)

            # Total time
            timings['total'] = time.time() - total_start

            # Display timing summary
            self._display_timing_summary(timings)

            print("\n" + "=" * 80)
            print("PROCESSING COMPLETE")
            print("=" * 80)

            # Return result - READY FOR API RESPONSE
            return {
                'success': True,
                'audio_info': audio_info,
                'segments': merged_segments,
                'statistics': statistics,
                'summary': summary,
                'timings': timings
            }

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            raise

    def _display_timing_summary(self, timings: Dict[str, float]):
        """Display detailed timing summary in console"""
        print("\n" + "=" * 80)
        print("TIMING BREAKDOWN")
        print("=" * 80)

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
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"{label:.<30} {duration:>6.2f}s [{bar}] {percentage:>5.1f}%")

        print("-" * 80)
        print(f"{'TOTAL TIME':.<30} {total:>6.2f}s")
        print("=" * 80)